import abc
from typing import Any, Generic, TypeVar

import torch

from sckitflow.core._data_utils import subscript_step_data
from sckitflow.core._types import PredictionData, StepData, TMatchFn, TNoiseSamplerFn, TTimeSamplerFn
from sckitflow.core.nn._modules import BaseModule
from sckitflow.core.probability_paths._probability_paths import BaseProbabilityPath, LinearDiracProbabilityPath

__all__ = [
    "ProtocolSpecs",
    "FlowSpecs",
    "BaseTrainingProtocol",
    "BaseFlowTrainingProtocol",
    "BaseInferenceProtocol",
    "BaseFlowInferenceProtocol",
    "BaseMatchingProtocol",
    "MatchingProtocol",
    "ProtocolMixin",
    "TrainingProtocolWrapper",
    "InferenceProtocolWrapper",
    "MatchedTrainingProtocol",
]


# -------------------- Initialization behaviors --------------------
class ProtocolSpecs:
    """Store for the protocol specifications.

    This class simply holds the necessary information required to
    define a protocol. It is used to instantiate both training and inference
    protocols.
    """

    def __init__(
        self,
        module: BaseModule,
        dtype: torch.dtype = torch.float32,
        device_id: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        """Initializes the protocol specifications with the given settings.

        :param module: An initialized neural module the protocol builds upon. It should be
            an initialized instance of a class inheriting from `BaseModule`.
        :param dtype: A `torch.dtype` object used to store the module weights.
        :param device_id: A string identifier of the device location for the module
            weights and input data.
        """
        self._dtype = dtype
        self._device_id = device_id
        self._module = module.to(device=self._device_id, dtype=self._dtype)

    def set_train_mode(self, mode: bool) -> None:
        """Sets the underlying module in training or inference mode.

        :param mode: When `True`, the neural module will be set to `train`.
            When `False`, its forward pass will be performed in evaluation mode.
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

    This class simply holds the necessary information required to
    define a flow model. It is used to instantiate both training and inference
    flow protocols.

    This class bases `ProtocolSpecs`; as additional arguments, it expects a
    probability path, a time sampler, a noise sampler and a boolean flag indicating
    whether the generation starts from noise.
    """

    def __init__(
        self,
        module: BaseModule,
        probability_path: BaseProbabilityPath | None = None,
        time_sampler: TTimeSamplerFn | None = None,
        noise_sampler: TNoiseSamplerFn | None = None,
        generate_from_noise: bool = False,
        dtype: torch.dtype = torch.float32,
        device_id: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        """Initializes the flow specifications.

        :param module: An initialized neural module the protocol builds upon. It should be
            an initialized instance of a class inheriting from `BaseModule`.
        :param probability_path: (Optional) An instance of `BaseProbabilityPath` used to define the
            tractable conditional probability path for the flow model. When `None`, the constructor
            will automatically initialize a `LinearDiracProbability`.
        :param time_sampler: (Optional) a callable to sample random time indices for the flow model.
            When `None`, it will be automatically initialized to a uniform distribution over [0, 1].
        :param noise_sampler: (Optional) a callable o sample random noise states as source for the flow model.
            It is only used when `generate_from_noise` is `True`, or when the data does not contain source states.
            When `None`, it will automatically set initialized to an isotropic Gaussian distribution.
        :param generate_from_noise: Boolean flag indicating whether the model interpolates from a tractable
            noise distribution, rather than from a control distribution. Defaults to `False`, in which case
            a source distribution is expected. When `True`, the interpolation will happen from noise, even
            when source states are present; the information on the source states will be injected as an
            extra conditioning in the neural module.
        :param dtype: A `torch.dtype` object used to store the module weights.
        :param device_id: A string identifier of the device location for the module
            weights and input data.
        """
        if generate_from_noise and noise_sampler is None:
            raise TypeError("When generating from noise you need to pass a noise sampler.")

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


# -------------------- Abstract Contracts (no storage) --------------------
class _AbstractTrainingProtocol(abc.ABC):
    """Pure abstract contract for training protocols.

    A training protocol is required to define the `compute_loss` method.
    """

    @abc.abstractmethod
    def compute_loss(self, step_data: StepData) -> tuple[torch.Tensor, dict[str, Any]]: ...


class _AbstractInferenceProtocol(abc.ABC):
    """Pure abstract contract for inference protocols.

    An inference protocol is required to define the `predict` method.
    """

    @abc.abstractmethod
    def predict(self, step_data: StepData) -> PredictionData: ...


class _AbstractMatchingProtocol(abc.ABC):
    """Pure abstract contract for matching protocols.

    A matching protocol is required to define the `predict` method.
    """

    @abc.abstractmethod
    def match(self, step_data: StepData) -> StepData: ...


# -------------------- Base Protocol Classes (still abstract) --------------------
class BaseTrainingProtocol(ProtocolSpecs, _AbstractTrainingProtocol):
    """Base training protocol, inheriting from both `ProtocolSpecs` and `_AbstractTrainingProtocol`"""

    ...


class BaseFlowTrainingProtocol(FlowSpecs, _AbstractTrainingProtocol):
    """Base flow training protocol, inheriting from both `FlowSpecs` and `_AbstractTrainingProtocol`"""

    ...


class BaseInferenceProtocol(ProtocolSpecs, _AbstractInferenceProtocol):
    """Base inference protocol, inheriting from both `ProtocolSpecs` and `_AbstractInferenceProtocol`"""

    ...


class BaseFlowInferenceProtocol(FlowSpecs, _AbstractInferenceProtocol):
    """Base flow inference protocol, inheriting from both `FlowSpecs` and `_AbstractInferenceProtocol`"""

    ...


class BaseMatchingProtocol(_AbstractMatchingProtocol):
    """Base class for matching protocols.

    Matching protocols are defined in terms of the `match_fn` callable, used to match source
    and target populations.
    """

    def __init__(self, match_fn: TMatchFn):
        """Initializes the matching protocol with the input `match_fn`.

        :param match_fn: A callable, satisfying the contract specified by `TMatchFn`,
            used to match source and target populations from a batch of data.
        """
        self._match_fn = match_fn

    @property
    def match_fn(self) -> TMatchFn:
        return self._match_fn


# -------------------- Matching protocol with callable --------------------
class MatchingProtocol(BaseMatchingProtocol):
    """Public matching protocol.

    Returns ``step_data`` unchanged when no source coupling data is present
    or when ``match_fn`` yields no indices; otherwise returns a subscripted
    copy aligned on the matched indices.
    """

    def match(
        self,
        step_data: StepData,
    ) -> StepData:
        """Matches the input state data using the underlying `match_fn`.

        When neither `source_coupling_lin` nor `source_coupling_quad`
        are present, it will return the step data unchanged. This will also
        be the case when the `match_fn` returns either `src_idxs` or `tgt_idxs`
        as `None` - no operation will be performed on the data

        :param step_data: The input state data to match.
        """
        # ---- Parse coupling data ----
        source_lin = step_data["source_coupling_lin"]
        source_quad = step_data["source_coupling_quad"]
        target_lin = step_data["target_coupling_lin"]
        target_quad = step_data["target_coupling_quad"]

        # ---- Early return when source is not provided ----
        if source_lin is None and source_quad is None:
            return step_data

        # ---- Match indices with callable ----
        src_idxs, tgt_idxs = self.match_fn(
            source_lin=source_lin,
            target_lin=target_lin,
            source_quad=source_quad,
            target_quad=target_quad,
        )

        # ---- No operation when indices are None ----
        if src_idxs is None or tgt_idxs is None:
            return step_data

        return subscript_step_data(step_data, src_idxs=src_idxs, tgt_idxs=tgt_idxs)


# -------------------- Wrapped protocols --------------------
_P = TypeVar("_P", bound=ProtocolSpecs)


class ProtocolMixin(Generic[_P]):
    """Shared delegation logic for protocol wrappers."""

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


# -------------------- Ready‑made Wrapper Classes --------------------
class TrainingProtocolWrapper(
    ProtocolMixin[BaseTrainingProtocol | BaseFlowTrainingProtocol], _AbstractTrainingProtocol
):
    """Concrete wrapper for a training protocol.

    Delegates `compute_loss` to the wrapped protocol.
    """

    def __init__(self, protocol: BaseTrainingProtocol | BaseFlowTrainingProtocol):
        """Initializes the wrapped training protocol from an underlying one.

        :param protocol: The base protocol to wrap around. It needs to be an instance of
            `BaseTrainingProtocol` or `BaseFlowTrainingProtocol`.
        """
        if not isinstance(protocol, BaseTrainingProtocol | BaseFlowTrainingProtocol):
            raise TypeError("Wrapped protocol must provide compute_loss.")
        super().__init__(protocol)

    def compute_loss(self, step_data: StepData) -> tuple[torch.Tensor, dict[str, Any]]:
        """Wraps around the `.compute_loss` call from underlying protocol."""
        return self._protocol.compute_loss(step_data)


class InferenceProtocolWrapper(
    ProtocolMixin[BaseInferenceProtocol | BaseFlowInferenceProtocol], _AbstractInferenceProtocol
):
    """Concrete wrapper for an inference protocol.

    Delegates `predict` to the wrapped protocol.
    """

    def __init__(self, protocol: BaseInferenceProtocol | BaseFlowInferenceProtocol):
        """Initializes the wrapped training protocol from an underlying one.

        :param protocol: The base protocol to wrap around. It needs to be an instance of
            `BaseInferenceProtocol` or `BaseFlowInferenceProtocol`.
        """
        if not isinstance(protocol, BaseInferenceProtocol | BaseFlowInferenceProtocol):
            raise TypeError("Wrapped protocol must provide predict.")
        super().__init__(protocol)

    def predict(self, step_data: StepData) -> PredictionData:
        """Wraps around the `.compute_loss` call from underlying protocol."""
        return self._protocol.predict(step_data)


class MatchedTrainingProtocol(TrainingProtocolWrapper):
    """Class for matched training protocols.

    Takes as input a training protocol (`protocol`) to wrap around and a matching callable (`match_fn`).
    The matching callable is used to instantiate the `self.matcher` attribute,
    a `MatchingProtocol`, defined in terms of the `match_fn`.

    The `.compute_loss` method of the wrapped protocol is called on the matched
    `step_data`; that is, the step data is first matched using `self.matcher.match(...)`, then the
    resulting matched data is passed to `self.protocol.compute_loss`.
    """

    def __init__(self, protocol: BaseTrainingProtocol | BaseFlowTrainingProtocol, match_fn: TMatchFn) -> None:
        """Initializes the matched training protocol.

        :param protocol: An instance of a `BaseTrainingProtocol` to wrap around;
            the wrapped class needs to implement the `.compute_loss` method.
        :param match_fn: The matching function, used to couple the batches.
            It needs to satisfy the contract specified by `TMatchFn`.
        """
        super().__init__(protocol)
        self._matcher = MatchingProtocol(match_fn)

    def compute_loss(self, step_data: StepData) -> tuple[torch.Tensor, dict[str, Any]]:
        """Wraps around the `self.protocol.compute_loss`, calling it on matched data."""
        matched = self._matcher.match(step_data)
        return self._protocol.compute_loss(matched)

    @property
    def matcher(self) -> MatchingProtocol:
        """The matcher used to pair the data."""
        return self._matcher
