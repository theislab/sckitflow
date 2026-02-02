from collections.abc import Callable, Sequence
from typing import Literal, Protocol

import numpy as np
import torch

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


class TConditioningFn(Protocol):
    def __call__(
        self,
        encoded_t: torch.Tensor,
        encoded_state: torch.Tensor,
        *args: torch.Tensor,
    ) -> torch.Tensor: ...
