"""Core method abstractions and shared implementation.

The two structural contracts (``SupportsTraining`` / ``SupportsInference``), their config families,
and the shared ``AbstractFlowMethod`` that stores a module, its shape spec, its probability path and
its time / noise samplers.
"""

import abc
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch
from scfit.registry import Component

from sckitflow._samplers import (
    IsotropicGaussianNoiseSamplerConfig,
    NoiseSamplerConfig,
    TimeSamplerConfig,
    UniformTimeSamplerConfig,
)
from sckitflow.core._data_utils import subscript_step_data
from sckitflow.core._types import MatchFn, PredictionData, SamplerFn, StepData
from sckitflow.core.probability_paths._config import ProbabilityPathConfig
from sckitflow.core.probability_paths._probability_paths import (
    BaseProbabilityPath,
    LinearDiracProbabilityPath,
)
from sckitflow.data._dims import DataDimensions

__all__ = [
    # The two contracts
    "SupportsTraining",
    "SupportsInference",
    # Code reuse for implementations -- not contracts.
    "AbstractMethod",
    "AbstractFlowMethod",
    "FlowMethodConfig",
    # Config families
    "TrainingMethodConfig",
    "InferenceMethodConfig",
    # Matching
    "BaseMatcher",
    "Matcher",
    # Matched training
    "MatchedTrainingMethod",
]


# -------------------- The two contracts --------------------
@runtime_checkable
class SupportsTraining(Protocol):
    """A module to train, its shape spec, plus `compute_loss`."""

    @property
    def module(self) -> torch.nn.Module: ...

    @property
    def dims(self) -> DataDimensions: ...

    def compute_loss(
        self, step_data: StepData, *, generator: torch.Generator, rng: np.random.Generator
    ) -> tuple[torch.Tensor, dict[str, Any]]: ...


@runtime_checkable
class SupportsInference(Protocol):
    """A module to run, its shape spec, plus `predict`."""

    @property
    def module(self) -> torch.nn.Module: ...

    @property
    def dims(self) -> DataDimensions: ...

    def predict(self, step_data: StepData, *, generator: torch.Generator) -> PredictionData: ...


class TrainingMethodConfig(Component):
    """Family base for anything that configures a training method."""

    def build(self, module: torch.nn.Module, *, dims: DataDimensions) -> SupportsTraining:
        """The training method around ``module``, for the given data shape."""
        raise NotImplementedError


class InferenceMethodConfig(Component):
    """Family base for anything that configures an inference method."""

    def build(self, module: torch.nn.Module, *, dims: DataDimensions) -> SupportsInference:
        """The inference method around ``module``, for the given data shape."""
        raise NotImplementedError


# -------------------- Shared implementation --------------------
def _placement_of(module: torch.nn.Module) -> tuple[torch.device, torch.dtype]:
    """Where a tensor this method allocates has to live, given its module.

    Read off the module's first parameter: Lightning places parameters, not the
    tensors a method draws, so the parameters are the reference for anything new.
    A parameterless module has no placement of its own -- and cannot be on a GPU,
    since there would be nothing to move -- so CPU / float32 is the default.

    Deliberately not cached: Lightning moves the module inside ``Trainer.fit`` /
    ``.predict``, after the method was constructed. Reading the parameter on each
    call is what keeps the placement in sync with the module's current state.
    """
    first = next(iter(module.parameters()), None)
    if first is None:
        return torch.device("cpu"), torch.float32
    return first.device, first.dtype


class AbstractMethod:
    """Holds the neural module a method is built on, and the shape spec it was built for.

    Purely for code reuse between implementations -- never type-test against this,
    use `SupportsTraining` / `SupportsInference`.

    Construction has no side effects: the module is stored as given, never moved
    or retyped, so handing one module to a training and an inference method is
    safe. Placement is the caller's -- ``module.to(device, dtype)`` before
    constructing, and ``module.train(mode)`` to switch modes. Inside
    `compute_loss` / `predict`, `dims` is the reference for *shape* and
    `device` / `dtype` (read live off the module) are the reference for
    *placement of anything the method allocates*.
    """

    def __init__(self, module: torch.nn.Module, *, dims: DataDimensions) -> None:
        """Keeps `module` and `dims` as given.

        :param module: An initialized `torch.nn.Module` the method builds upon,
            already on the device and dtype you want to run in.
        :param dims: The shape spec of the data the module was built for.
            ``state_dim`` sizes the latent; the covariate and coupling dims
            describe what the model expects on the conditioning side.
        """
        self._module = module
        self._dims = dims

    @property
    def module(self) -> torch.nn.Module:
        return self._module

    @property
    def dims(self) -> DataDimensions:
        return self._dims

    @property
    def device(self) -> torch.device:
        """Where a tensor this method allocates has to live; see :func:`_placement_of`."""
        return _placement_of(self.module)[0]

    @property
    def dtype(self) -> torch.dtype:
        """The dtype a tensor this method allocates has to have; see :func:`_placement_of`."""
        return _placement_of(self.module)[1]


class FlowMethodConfig(Component):
    """The flow configuration every flow method shares; each method's config extends it.

    Only configs are accepted, never live objects or raw callables, so ``Model.save_pretrained`` can
    write the spec and ``Model.from_pretrained`` rebuilds the same method from it.
    """

    probability_path: ProbabilityPathConfig | None = None
    """A probability-path config. ``None`` uses ``LinearDiracProbabilityPath``."""

    time_sampler: TimeSamplerConfig | None = None
    """A time-sampler config. ``None`` uses ``UniformTimeSamplerConfig``."""

    noise_sampler: NoiseSamplerConfig | None = None
    """A noise-sampler config. ``None`` uses ``IsotropicGaussianNoiseSamplerConfig``."""

    generate_from_noise: bool = False
    """Interpolate from noise even when source states are present; the source is then extra conditioning."""


class AbstractFlowMethod(AbstractMethod):
    """Adds the flow configuration that flow trainers and flow predictors share.

    Its settings come from one config object, an `FlowMethodConfig` or a
    subclass of it. A training and an inference method given the same module and
    flow settings share the path, samplers and module. The data shape comes from
    `dims`; it is not carried on the config, because it is a property of the run's
    data, not of the flow settings.
    """

    def __init__(self, module: torch.nn.Module, config: FlowMethodConfig, *, dims: DataDimensions) -> None:
        """Builds the path and samplers from ``config``.

        :param module: An initialized neural module the method builds upon.
        :param config: The flow settings, the method's own config.
        :param dims: The shape spec of the data the module was built for.
        """
        super().__init__(module, dims=dims)
        path_cfg = config.probability_path
        self._probability_path = LinearDiracProbabilityPath() if path_cfg is None else path_cfg.build()
        self._time_sampler = (config.time_sampler or UniformTimeSamplerConfig()).build()
        self._noise_sampler = (config.noise_sampler or IsotropicGaussianNoiseSamplerConfig()).build()
        self._generate_from_noise = config.generate_from_noise
        if dims.source_state_dim is not None and not self._generate_from_noise:
            raise ValueError(
                f"The sources have their own state rep (`source_sample_rep`, {dims.source_state_dim} features; the "
                f"targets' has {dims.state_dim}), so the flow cannot start from them (`generate_from_noise=False`). "
                "Generate from noise conditioned on the encoded source instead, as GENOT does: set "
                "`generate_from_noise=True` and give the velocity field a source encoder (`source_encoder_mlp_kwargs`)."
            )

    @property
    def probability_path(self) -> BaseProbabilityPath:
        return self._probability_path

    @property
    def noise_sampler(self) -> SamplerFn:
        return self._noise_sampler

    @property
    def time_sampler(self) -> SamplerFn:
        return self._time_sampler

    @property
    def generate_from_noise(self) -> bool:
        return self._generate_from_noise


# -------------------- Matching --------------------
class BaseMatcher(abc.ABC):
    """Base class for matching methods.

    Stores the `match_fn` callable used to match source and target populations.
    """

    def __init__(self, match_fn: MatchFn) -> None:
        """Initializes the matching method.

        :param match_fn: A callable satisfying `MatchFn`, used to match source
            and target populations from a batch of data.
        """
        self._match_fn = match_fn

    @abc.abstractmethod
    def match(self, step_data: StepData, *, rng: np.random.Generator) -> StepData: ...

    @property
    def match_fn(self) -> MatchFn:
        return self._match_fn


class Matcher(BaseMatcher):
    """Public matching method.

    Returns ``step_data`` unchanged when no source coupling data is present
    or when ``match_fn`` yields no indices; otherwise returns a subscripted
    copy aligned on the matched indices.
    """

    def match(self, step_data: StepData, *, rng: np.random.Generator) -> StepData:
        """Matches the input state data using the underlying `match_fn`, drawing from ``rng``.

        Returns `step_data` unchanged when neither `source_coupling_lin` nor
        `source_coupling_quad` are present, or when `match_fn` returns either
        `src_idxs` or `tgt_idxs` as `None`.
        """
        source_lin = step_data["source_coupling_lin"]
        source_quad = step_data["source_coupling_quad"]
        target_lin = step_data["target_coupling_lin"]
        target_quad = step_data["target_coupling_quad"]

        if source_lin is None and source_quad is None:
            return step_data

        src_idxs, tgt_idxs = self.match_fn(
            source_lin=source_lin,
            target_lin=target_lin,
            source_quad=source_quad,
            target_quad=target_quad,
            rng=rng,
        )

        if src_idxs is None or tgt_idxs is None:
            return step_data

        return subscript_step_data(step_data, src_idxs=src_idxs, tgt_idxs=tgt_idxs)


# -------------------- Matched training --------------------
class MatchedTrainingMethod:
    """Runs a matcher over the batch, then delegates to the wrapped training method.

    Satisfies `SupportsTraining` structurally, so it is usable anywhere a plain
    training method is -- including wrapped again.
    """

    def __init__(self, method: SupportsTraining, matcher: BaseMatcher) -> None:
        """Initializes the matched training method.

        :param method: The training method to wrap around.
        :param matcher: The matcher pairing source and target, e.g.
            ``Matcher(match_fn)``. Taken rather than built, so a `BaseMatcher`
            subclass can be used in its place.
        """
        self._method = method
        self._matcher = matcher

    def compute_loss(
        self, step_data: StepData, *, generator: torch.Generator, rng: np.random.Generator
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        matched = self._matcher.match(step_data, rng=rng)
        return self._method.compute_loss(matched, generator=generator, rng=rng)

    @property
    def method(self) -> SupportsTraining:
        """The wrapped training method."""
        return self._method

    @property
    def matcher(self) -> BaseMatcher:
        """The matcher used to pair the data."""
        return self._matcher

    @property
    def module(self) -> torch.nn.Module:
        """The wrapped method's module; `SupportsTraining` requires it."""
        return self._method.module

    @property
    def dims(self) -> DataDimensions:
        """The wrapped method's shape spec; `SupportsTraining` requires it."""
        return self._method.dims

    @property
    def device(self) -> torch.device:
        """The wrapped method's placement; forwarded so any caller reaching for it sees the same one."""
        return self._method.device  # type: ignore[attr-defined]

    @property
    def dtype(self) -> torch.dtype:
        """The wrapped method's dtype; forwarded for the same reason as ``device``."""
        return self._method.dtype  # type: ignore[attr-defined]
