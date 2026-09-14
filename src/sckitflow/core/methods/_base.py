import abc
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

import torch

from sckitflow.core._data_utils import subscript_step_data
from sckitflow.core._types import PredictionData, StepData, TMatchFn, TNoiseSamplerFn, TTimeSamplerFn
from sckitflow.core.nn._modules import BaseModule
from sckitflow.core.probability_paths._probability_paths import BaseProbabilityPath, LinearDiracProbabilityPath

__all__ = [
    # Structural contracts
    "SupportsProtocol",
    "SupportsTraining",
    "SupportsInference",
    # Storage
    "ProtocolSpecs",
    "FlowSpecs",
    # Base protocols
    "BaseTrainingProtocol",
    "BaseFlowTrainingProtocol",
    "BaseInferenceProtocol",
    "BaseFlowInferenceProtocol",
    "BaseMatchingProtocol",
    "MatchingProtocol",
    # Wrappers
    "ProtocolMixin",
    "TrainingProtocolWrapper",
    "InferenceProtocolWrapper",
    "MatchedTrainingProtocol",
]


# -------------------- Structural contracts --------------------
@runtime_checkable
class SupportsProtocol(Protocol):
    """Anything that exposes the protocol's storage surface.

    Satisfied structurally by `ProtocolSpecs`, `FlowSpecs`, every `_SpecsHolder`
    subclass, and every wrapper built on top of them.
    """

    @property
    def module(self) -> BaseModule: ...
    @property
    def dtype(self) -> torch.dtype: ...
    @property
    def device_id(self) -> str: ...
    def set_train_mode(self, mode: bool) -> None: ...


@runtime_checkable
class SupportsTraining(SupportsProtocol, Protocol):
    """Storage plus `compute_loss`."""

    def compute_loss(self, step_data: StepData) -> tuple[torch.Tensor, dict[str, Any]]: ...


@runtime_checkable
class SupportsInference(SupportsProtocol, Protocol):
    """Storage plus `predict`."""

    def predict(self, step_data: StepData) -> PredictionData: ...


# -------------------- Storage mixins --------------------
class ProtocolSpecs:
    """Store for the protocol specifications.

    Holds the neural module and the associated dtype/device configuration.
    A single instance can be shared by any number of protocols, so the module,
    dtype, and device stay in sync across them.
    """

    def __init__(
        self,
        module: BaseModule,
        dtype: torch.dtype | None = None,
        device_id: str | None = None,
    ) -> None:
        """Initializes the protocol specifications with the given settings.

        :param module: An initialized neural module the protocol builds upon.
            It should be an initialized instance of a class inheriting from
            `BaseModule`.
        :param dtype: A `torch.dtype` object used to store the module weights.
        :param device_id: A string identifier of the device location for the
            module weights and input data.
        """
        self._dtype = torch.float32 if dtype is None else dtype
        if device_id is None:
            device_id = "cuda" if torch.cuda.is_available() else "cpu"
        self._device_id = device_id
        self._module = module.to(device=self._device_id, dtype=self._dtype)

    def set_train_mode(self, mode: bool) -> None:
        """Sets the underlying module in training or inference mode.

        :param mode: When `True`, the neural module is set to `train`. When
            `False`, its forward pass is performed in evaluation mode.
        """
        if mode:
            self.module.train()
        else:
            self.module.eval()

    @property
    def module(self) -> BaseModule:
        return self._module

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def device_id(self) -> str:
        return self._device_id


class FlowSpecs(ProtocolSpecs):
    """Store for the flow specifications.

    Extends `ProtocolSpecs` with a probability path, a time sampler, a noise
    sampler, and a flag indicating whether generation starts from noise. A
    single instance can be shared by a flow training protocol and a flow
    inference protocol, so both see the same path and samplers.
    """

    def __init__(
        self,
        module: BaseModule,
        probability_path: BaseProbabilityPath | None = None,
        time_sampler: TTimeSamplerFn | None = None,
        noise_sampler: TNoiseSamplerFn | None = None,
        generate_from_noise: bool = False,
        dtype: torch.dtype | None = None,
        device_id: str | None = None,
    ) -> None:
        """Initializes the flow specifications.

        :param module: An initialized neural module the protocol builds upon.
        :param probability_path: Optional `BaseProbabilityPath`. Defaults to a
            `LinearDiracProbabilityPath`.
        :param time_sampler: Optional callable sampling times in [0, 1].
            Defaults to `torch.rand`.
        :param noise_sampler: Optional callable sampling source noise.
            Defaults to `torch.randn`.
        :param generate_from_noise: When `True`, interpolation starts from the
            noise distribution even if source states are present (source
            information is passed as extra conditioning instead).
        :param dtype: A `torch.dtype` object used to store the module weights.
        :param device_id: A string identifier of the device location.
        """
        super().__init__(module, dtype=dtype, device_id=device_id)

        self._probability_path = LinearDiracProbabilityPath() if probability_path is None else probability_path
        self._noise_sampler = torch.randn if noise_sampler is None else noise_sampler
        self._time_sampler = torch.rand if time_sampler is None else time_sampler
        self._generate_from_noise = generate_from_noise

    @property
    def probability_path(self) -> BaseProbabilityPath:
        return self._probability_path

    @property
    def noise_sampler(self) -> TNoiseSamplerFn | None:
        return self._noise_sampler

    @property
    def time_sampler(self) -> TTimeSamplerFn:
        return self._time_sampler

    @property
    def generate_from_noise(self) -> bool:
        return self._generate_from_noise


# -------------------- Specs holders --------------------
# The protocols below do not *inherit* from `ProtocolSpecs` / `FlowSpecs`;
# they hold an instance and delegate to it. This lets a training and an
# inference protocol share a single specs instance, so the module, dtype,
# device, and flow configuration are guaranteed identical.
_S = TypeVar("_S", bound=ProtocolSpecs)


class _SpecsHolder(Generic[_S]):
    """Delegates the storage surface to a shared `ProtocolSpecs` instance."""

    def __init__(self, specs: _S) -> None:
        self._specs: _S = specs

    @property
    def specs(self) -> _S:
        return self._specs

    @property
    def module(self) -> BaseModule:
        return self._specs.module

    @property
    def dtype(self) -> torch.dtype:
        return self._specs.dtype

    @property
    def device_id(self) -> str:
        return self._specs.device_id

    def set_train_mode(self, mode: bool) -> None:
        self._specs.set_train_mode(mode)


class _FlowSpecsHolder(_SpecsHolder[FlowSpecs]):
    """Adds flow-specific delegation on top of the storage surface."""

    @property
    def probability_path(self) -> BaseProbabilityPath:
        return self._specs.probability_path

    @property
    def time_sampler(self) -> TTimeSamplerFn:
        return self._specs.time_sampler

    @property
    def noise_sampler(self) -> TNoiseSamplerFn | None:
        return self._specs.noise_sampler

    @property
    def generate_from_noise(self) -> bool:
        return self._specs.generate_from_noise


# -------------------- Base protocol classes --------------------
class BaseTrainingProtocol(_SpecsHolder[ProtocolSpecs], abc.ABC):
    """Base training protocol: shared storage + `compute_loss` contract.

    Constructed with a `ProtocolSpecs` instance, which can be shared with an
    inference protocol so the module, dtype, and device stay in sync.
    """

    @abc.abstractmethod
    def compute_loss(self, step_data: StepData) -> tuple[torch.Tensor, dict[str, Any]]: ...


class BaseFlowTrainingProtocol(_FlowSpecsHolder, abc.ABC):
    """Base flow training protocol: shared `FlowSpecs` + `compute_loss` contract."""

    @abc.abstractmethod
    def compute_loss(self, step_data: StepData) -> tuple[torch.Tensor, dict[str, Any]]: ...


class BaseInferenceProtocol(_SpecsHolder[ProtocolSpecs], abc.ABC):
    """Base inference protocol: shared storage + `predict` contract."""

    @abc.abstractmethod
    def predict(self, step_data: StepData) -> PredictionData: ...


class BaseFlowInferenceProtocol(_FlowSpecsHolder, abc.ABC):
    """Base flow inference protocol: shared `FlowSpecs` + `predict` contract."""

    @abc.abstractmethod
    def predict(self, step_data: StepData) -> PredictionData: ...


class BaseMatchingProtocol(abc.ABC):
    """Base class for matching protocols.

    Stores the `match_fn` callable used to match source and target populations.
    """

    def __init__(self, match_fn: TMatchFn) -> None:
        """Initializes the matching protocol.

        :param match_fn: A callable satisfying `TMatchFn`, used to match source
            and target populations from a batch of data.
        """
        self._match_fn = match_fn

    @abc.abstractmethod
    def match(self, step_data: StepData) -> StepData: ...

    @property
    def match_fn(self) -> TMatchFn:
        return self._match_fn


class MatchingProtocol(BaseMatchingProtocol):
    """Public matching protocol.

    Returns ``step_data`` unchanged when no source coupling data is present
    or when ``match_fn`` yields no indices; otherwise returns a subscripted
    copy aligned on the matched indices.
    """

    def match(self, step_data: StepData) -> StepData:
        """Matches the input state data using the underlying `match_fn`.

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
        )

        if src_idxs is None or tgt_idxs is None:
            return step_data

        return subscript_step_data(step_data, src_idxs=src_idxs, tgt_idxs=tgt_idxs)


# -------------------- Wrappers --------------------
_P = TypeVar("_P", bound=SupportsProtocol)


class ProtocolMixin(Generic[_P]):
    """Shared delegation logic for protocol wrappers.

    Delegates the storage surface (`module`, `dtype`, `device_id`,
    `set_train_mode`) to the wrapped protocol. The bound is the structural
    `SupportsProtocol`, so wrappers do not require a shared base class with
    what they wrap.
    """

    def __init__(self, protocol: _P) -> None:
        self._protocol: _P = protocol

    def set_train_mode(self, mode: bool) -> None:
        self._protocol.set_train_mode(mode)

    @property
    def protocol(self) -> _P:
        return self._protocol

    @property
    def dtype(self) -> torch.dtype:
        return self._protocol.dtype

    @property
    def device_id(self) -> str:
        return self._protocol.device_id

    @property
    def module(self) -> BaseModule:
        return self._protocol.module


class TrainingProtocolWrapper(ProtocolMixin[SupportsTraining]):
    """Concrete wrapper for a training protocol.

    Accepts anything satisfying `SupportsTraining` (base, flow, already
    wrapped, or user-defined) and delegates `compute_loss` to it.
    """

    def __init__(self, protocol: SupportsTraining) -> None:
        # `isinstance` on a runtime_checkable protocol verifies attribute presence,
        # not signature. Bad signatures surface at the first `compute_loss` call.
        if not isinstance(protocol, SupportsTraining):
            raise TypeError("Wrapped protocol must provide compute_loss.")
        super().__init__(protocol)

    def compute_loss(self, step_data: StepData) -> tuple[torch.Tensor, dict[str, Any]]:
        return self._protocol.compute_loss(step_data)


class InferenceProtocolWrapper(ProtocolMixin[SupportsInference]):
    """Concrete wrapper for an inference protocol.

    Accepts anything satisfying `SupportsInference` and delegates `predict`.
    """

    def __init__(self, protocol: SupportsInference) -> None:
        if not isinstance(protocol, SupportsInference):
            raise TypeError("Wrapped protocol must provide predict.")
        super().__init__(protocol)

    def predict(self, step_data: StepData) -> PredictionData:
        return self._protocol.predict(step_data)


class MatchedTrainingProtocol(TrainingProtocolWrapper):
    """Matched training protocol: wraps a training protocol + a `match_fn`.

    The wrapped protocol's `compute_loss` is called on `step_data` after it has
    been matched by an internal `MatchingProtocol`.
    """

    def __init__(self, protocol: SupportsTraining, match_fn: TMatchFn) -> None:
        """Initializes the matched training protocol.

        :param protocol: The training protocol to wrap around.
        :param match_fn: The matching function satisfying `TMatchFn`.
        """
        super().__init__(protocol)
        self._matcher = MatchingProtocol(match_fn)

    def compute_loss(self, step_data: StepData) -> tuple[torch.Tensor, dict[str, Any]]:
        matched = self._matcher.match(step_data)
        return self._protocol.compute_loss(matched)

    @property
    def matcher(self) -> MatchingProtocol:
        """The matcher used to pair the data."""
        return self._matcher
