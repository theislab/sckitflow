from collections.abc import Callable, Sequence
from typing import Protocol

import numpy as np
import torch

ShapeLike = Sequence[int] | torch.Size

TensorLike = torch.Tensor | np.ndarray

MappedTensor = dict[str, torch.Tensor]

TVfFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

TTimeFeaturesFn = Callable[[torch.Tensor, int], torch.Tensor]

TMeanFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
TDriftFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
TSigmaFn = Callable[[torch.Tensor], torch.Tensor]


class TConditioningFn(Protocol):
    def __call__(
        self,
        encoded_t: torch.Tensor,
        encoded_state: torch.Tensor,
        *args: torch.Tensor,
    ) -> torch.Tensor: ...
