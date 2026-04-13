from collections.abc import Callable, Sequence
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
TMatchFn = Callable[[TensorLike, TensorLike], MatchFnOut]

TTimeSamplerFn = Callable[[tuple[int, ...]], TensorLike] | Callable[[tuple[int, ...]], tuple[TensorLike, TensorLike]]
TNoiseSamplerFn = Callable[[tuple[int, ...]], TensorLike]


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
