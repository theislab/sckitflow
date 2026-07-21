from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, Protocol, TypeVar

import numpy as np
import torch
from torch import Tensor

if TYPE_CHECKING:
    from sc_flow.flow._vf import BaseVelocityField

try:
    from numpy.typing import NDArray

    NumpyArray = NDArray[np.float32 | np.float64]
except (ImportError, TypeError):
    NumpyArray = np.ndarray

from sc_flow._types import PredictionData

ShapeLike = Sequence[int] | torch.Size

TensorLike = torch.Tensor | np.ndarray

MappedTensor = dict[str, torch.Tensor]


@dataclass
class StepData:
    """Batch payload a train/predict step consumes: matched source/target tensors.

    Canonical home for the step input contract. It is shared by the training
    methods and by consumers that call a trained model directly (e.g. the
    inverse-problem surrogate wrappers), so it lives in ``_types`` rather than
    inside ``methods/``.
    """

    target_state: torch.Tensor
    target_coupling_lin: torch.Tensor | None
    target_coupling_quad: torch.Tensor | None
    target_condition_data: Any | None
    target_group_data: Any | None
    source_state: torch.Tensor | None
    source_coupling_lin: torch.Tensor | None
    source_coupling_quad: torch.Tensor | None
    source_condition_data: Any | None
    source_group_data: Any | None


TVfFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

TTimeFeaturesFn = Callable[[torch.Tensor, int], torch.Tensor]

TMeanFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
TDriftFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
TSigmaFn = Callable[[torch.Tensor], torch.Tensor]
ScaleMethod = Literal["mean", "max", "median"] | float
LinCouplingMethod = Literal["exact", "sinkhorn", "partial", "unbalanced"] | None
QuadCouplingMethod = Literal["entropic_gromov_wasserstein", "entropic_fused_gromov_wasserstein"] | None
CostFN = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

MatchFnOut = tuple[TensorLike, TensorLike] | tuple[TensorLike, TensorLike, TensorLike]


class TMatchFn(Protocol):
    def __call__(
        self,
        source_lin: TensorLike,
        target_lin: TensorLike,
        **kwargs: Any,
    ) -> MatchFnOut: ...


class TSamplerFn(Protocol):
    def __call__(
        self,
        *size: int,
        **kwargs: Any,
    ) -> TensorLike | tuple[TensorLike, TensorLike]: ...


class TTimeSamplerFn(TSamplerFn):
    pass


class TNoiseSamplerFn(TSamplerFn):
    pass


class TConditioningFn(Protocol):
    def __call__(
        self,
        encoded_t: torch.Tensor,
        encoded_state: torch.Tensor,
        *args: torch.Tensor,
    ) -> torch.Tensor: ...


TDevice = str | torch.device
TNoiseType = Literal["scalar", "diagonal", "general", "additive"]
TSDEType = Literal["ito", "stratonovich"]


TODEDynamics = TypeVar("TODEDynamics", bound="BaseVelocityField")
TTimeStateDiffusion = Callable[[Tensor, Tensor], Tensor]
TTimeDiffusion = Callable[[Tensor], Tensor]
TDiffusion = TTimeDiffusion | TTimeStateDiffusion
TSDEDynamics = tuple[TODEDynamics, TDiffusion]
TSolverDynamics = TypeVar("TSolverDynamics", TODEDynamics, TSDEDynamics)


class SolverConfig(NamedTuple):
    """Configuration extracted from solver parameters and kwargs."""

    source_on_device: Tensor
    time_on_device: Tensor
    remaining_kwargs: dict[str, Any]


@dataclass
class PredictionData(PredictionData):
    """Stores prediction data for Flow Models

    * X (B, D)
    * raw_samples (N, B, D) or None
    * traj (T, 1, B, D) or (T, N, B, D) or None. The second case only when samples is not None.
    """

    X: torch.Tensor
    raw_samples: torch.Tensor | None = None
    traj: torch.Tensor | None = None

    @classmethod
    def concatenate(cls, preds: Collection["PredictionData"]) -> "PredictionData":
        # ----  Early return if empty collection ----
        if len(preds) == 0:
            raise ValueError("Cannot concatenate empty collection")

        # ---- Sanity check as they should all contain the same fields ----
        for idx, p in enumerate(preds):
            # ---- Get reference values from first element ----
            if idx == 0:
                ref_has_raw_samples = p.has_raw_samples
                ref_has_traj = p.has_traj

            # ---- Check that the raw samples match ---
            if p.has_raw_samples != ref_has_raw_samples:
                raise ValueError("All elements should have the same type of `raw_samples` attribute.")

            if p.has_traj != ref_has_traj:
                raise ValueError("All elements should have the same type of `traj`attribute.")

        # ---- Concatenate X, always on first dimension ----
        X = torch.cat([p.X for p in preds], dim=0)

        # --- Concatenate samples when provided ----
        raw_samples = [p.raw_samples for p in preds if p.raw_samples is not None]
        if raw_samples:
            raw_samples = torch.cat(raw_samples, dim=1)  # assuming [N, B, D] -> concat on B
        else:
            raw_samples = None

        # --- Concatenate trajectory when provided ----
        trajs = [p.traj for p in preds if p.traj is not None]
        if trajs:
            traj = torch.cat(trajs, dim=2)  # assuming [T, N, B, D] -> concat on B
        else:
            traj = None
        return cls(X=X, raw_samples=raw_samples, traj=traj)

    @property
    def has_raw_samples(self) -> bool:
        return self.raw_samples is not None

    @property
    def has_traj(self) -> bool:
        return self.traj is not None
