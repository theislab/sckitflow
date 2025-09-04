from collections.abc import Callable, Sequence

import torch

ShapeLike = Sequence[int] | torch.Size

VfFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

TTimeFeatursFn = Callable[
    [
        torch.Tensor,
    ],
    torch.Tensor,
]

TMeanFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
TDriftFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
TSigmaFn = Callable[[torch.Tensor], torch.Tensor]
