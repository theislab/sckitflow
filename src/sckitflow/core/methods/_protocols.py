import abc
from typing import Any, Generic, TypeVar

import torch

from sckitflow.core._data_utils import subscript_step_data
from sckitflow.core._types import PredictionData, StepData, TMatchFn
from sckitflow.core.nn._modules import BaseModule

__all__ = [
    "ProtocolSpecs",
    "MatchingSpecs",
    "MatchedProtocolSpecs",
    "BaseTrainingProtocol",
    "BaseInferenceProtocol",
    "BaseMethod",
    "BaseMatchingProtocol",
    "MatchingProtocol",
    "ProtocolMixin",
    "TrainingProtocolWrapper",
    "InferenceProtocolWrapper",
    "MethodWrapper",
    "MatchedTrainingProtocol",
]


# -------------------- Abstract Contracts (no storage) --------------------
class _AbstractTrainingProtocol(abc.ABC):
    """Pure abstract contract for training protocols."""

    @abc.abstractmethod
    def train_step(self, step_data: StepData) -> tuple[torch.Tensor, dict[str, Any]]: ...


class _AbstractInferenceProtocol(abc.ABC):
    """Pure abstract contract for inference protocols."""

    @abc.abstractmethod
    def predict(self, step_data: StepData) -> PredictionData: ...


class _AbstractMatchingProtocol(abc.ABC):
    """Pure abstract contract for matching protocols."""

    @abc.abstractmethod
    def match(self, step_data: StepData) -> StepData: ...


class _AbstractMethod(_AbstractTrainingProtocol, _AbstractInferenceProtocol):
    """Combined contract for full training + inference protocols."""

    pass


# -------------------- Storage Base --------------------
class ProtocolSpecs:
    """Mixin providing storage for module, dtype, and device.

    This class does **not** inherit from any abstract protocol; it only holds
    the neural module and associated properties. Concrete protocol classes
    combine this with the appropriate abstract contracts.
    """

    def __init__(
        self,
        module: BaseModule,
        dtype: torch.dtype = torch.float32,
        device_id: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        self._dtype = dtype
        self._device_id = device_id
        self._module = module.to(dtype=self._dtype, device=self._device_id)

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def device_id(self) -> str:
        return self._device_id

    @property
    def module(self) -> BaseModule:
        return self._module

    def set_train_mode(self, mode: bool) -> None:
        """Set the underlying module to training or evaluation mode."""
        if mode:
            self.module.train()
        else:
            self.module.eval()


class MatchingSpecs:
    """Mixin to store the information for the matching.

    The only information that is required is the `match_fn` callable,
    used to match the step data on its predefined coupling fields.
    """

    def __init__(self, match_fn: TMatchFn):
        self._match_fn: TMatchFn = match_fn

    @property
    def match_fn(self) -> TMatchFn:
        return self._match_fn


class MatchedProtocolSpecs(ProtocolSpecs, MatchingSpecs):
    """Mixin to jointly store matching and protocol information."""

    def __init__(
        self,
        module: BaseModule,
        match_fn: TMatchFn,
        dtype: torch.dtype = torch.float32,
        device_id: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        ProtocolSpecs.__init__(self, module, dtype=dtype, device_id=device_id)
        MatchingSpecs.__init__(self, match_fn)


# -------------------- Base Protocol Classes (still abstract) --------------------
class BaseTrainingProtocol(ProtocolSpecs, _AbstractTrainingProtocol):
    """Base class for training‑only protocols.

    Subclass this and implement `train_step`. The module, dtype, and device are
    stored automatically.
    """

    pass


class BaseInferenceProtocol(ProtocolSpecs, _AbstractInferenceProtocol):
    """Base class for inference‑only protocols.

    Subclass this and implement `predict`. The module, dtype, and device are
    stored automatically.
    """

    pass


class BaseMethod(ProtocolSpecs, _AbstractMethod):
    """Base class for full protocols (training + inference).

    Subclass this and implement both `train_step` and `predict`.
    """

    pass


class BaseMatchingProtocol(MatchingSpecs, _AbstractMatchingProtocol):
    """Base class for matching protocols.

    Subclass this and implement `match`. The matching function is stored
    automatically.
    """

    pass


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
class TrainingProtocolWrapper(ProtocolMixin[BaseTrainingProtocol], _AbstractTrainingProtocol):
    """Concrete wrapper for a training protocol.

    Delegates `train_step` to the wrapped protocol.
    """

    def __init__(self, protocol: BaseTrainingProtocol):
        if not isinstance(protocol, BaseTrainingProtocol):
            raise TypeError("Wrapped protocol must provide train_step.")
        super().__init__(protocol)

    def train_step(self, step_data: StepData) -> tuple[torch.Tensor, dict[str, Any]]:
        return self._protocol.train_step(step_data)


class InferenceProtocolWrapper(ProtocolMixin[BaseInferenceProtocol], _AbstractInferenceProtocol):
    """Concrete wrapper for an inference protocol.

    Delegates `predict` to the wrapped protocol.
    """

    def __init__(self, protocol: BaseInferenceProtocol):
        if not isinstance(protocol, BaseInferenceProtocol):
            raise TypeError("Wrapped protocol must provide predict.")
        super().__init__(protocol)

    def predict(self, step_data: StepData) -> PredictionData:
        return self._protocol.predict(step_data)


class MethodWrapper(ProtocolMixin[BaseMethod], _AbstractMethod):
    """Concrete wrapper for a full protocol (training + inference).

    Delegates both `train_step` and `predict` to the wrapped protocol.
    """

    def __init__(self, protocol: BaseMethod):
        if not isinstance(protocol, BaseMethod):
            raise TypeError("Wrapped protocol must provide both train_step and predict.")
        super().__init__(protocol)

    def train_step(self, step_data: StepData) -> tuple[torch.Tensor, dict[str, Any]]:
        return self._protocol.train_step(step_data)

    def predict(self, step_data: StepData) -> PredictionData:
        return self._protocol.predict(step_data)


class MatchedTrainingProtocol(TrainingProtocolWrapper):
    """Class for matched training protocols.

    Takes as input a training protocol (`protocol`) to wrap around and a matching callable (`match_fn`).
    The matching callable is used to instantiate the `self.matcher` attribute,
    a `MatchingProtocol`, defined in terms of the `match_fn`.

    The `.train_step` method of the wrapped protocol is called on the matched
    `step_data`; that is, the step data is first matched using `self.matcher.match(...)`, then the
    resulting matched data is passed to `self.protocol.train_step`.
    """

    def __init__(self, protocol: BaseTrainingProtocol, match_fn: TMatchFn) -> None:
        """Initializes the matched training protocol.

        :param protocol: An instance of a `BaseTrainingProtocol` to wrap around;
            the wrapped class needs to implement the `.train_step` method.
        :param match_fn: The matching function, used to couple the batches.
            It needs to satisfy the contract specified by `TMatchFn`.
        """
        super().__init__(protocol)
        self._matcher = MatchingProtocol(match_fn)

    def train_step(self, step_data: StepData) -> tuple[torch.Tensor, dict[str, Any]]:
        """Wraps around the `self.protocol.train_step`, calling it on matched data."""
        matched = self._matcher.match(step_data)
        return self._protocol.train_step(matched)

    @property
    def matcher(self) -> MatchingProtocol:
        """The matcher used to pair the data."""
        return self._matcher
