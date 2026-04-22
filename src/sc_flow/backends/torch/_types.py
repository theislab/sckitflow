from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, Protocol, TypeVar

import numpy as np
import torch
from torch import Tensor

if TYPE_CHECKING:
    from sc_flow.backends.torch.nn._vf import BaseVelocityField

try:
    from numpy.typing import NDArray

    NumpyArray = NDArray[np.float32 | np.float64]
except (ImportError, TypeError):
    NumpyArray = np.ndarray

ShapeLike = Sequence[int] | torch.Size

TensorLike = torch.Tensor | np.ndarray

MappedTensor = dict[str, torch.Tensor]

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


class TTimeSamplerFn(Protocol):
    def __call__(
        self,
        *size: int,
        **kwargs: Any,
    ) -> TensorLike | tuple[TensorLike, TensorLike]: ...


class TNoiseSamplerFn(Protocol):
    def __call__(
        self,
        reference: TensorLike,
        **kwargs: Any,
    ) -> TensorLike: ...


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
class PredictionData:
    samples: torch.Tensor
    traj: torch.Tensor | None = None

    @classmethod
    def concatenate(cls, preds: Collection["PredictionData"]) -> "PredictionData":
        samples = torch.cat([p.samples for p in preds], dim=0)
        trajs = [p.traj for p in preds if p.traj is not None]
        if trajs:
            traj = torch.cat(trajs, dim=1)  # assuming [T, N, D] -> concat on N
        else:
            traj = None
        return cls(samples=samples, traj=traj)
