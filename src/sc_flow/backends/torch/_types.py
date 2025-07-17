import sys
from collections.abc import Callable, Sequence

if sys.version_info < (3, 11):
    from typing_extensions import NotRequired, TypedDict
else:
    from typing import NotRequired, TypedDict

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


class ProbabilityPathDict(TypedDict):
    """"""  # noqa

    MU_T_FN_KEY: TMeanFn
    U_T_FN_KEY: TDriftFn
    SIGMA_T_FN_KEY: NotRequired[TSigmaFn]
