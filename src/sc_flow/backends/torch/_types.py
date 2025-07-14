import sys
from collections.abc import Callable, Sequence
from typing import Literal

if sys.version_info < (3, 11):
    from typing_extensions import NotRequired, TypedDict
else:
    from typing import NotRequired, TypedDict

import torch

ShapeLike = Sequence[int] | torch.Size

VfFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

TimeFeaturesId = Literal["ott-jax", "torch-cfm"]
TTimeFeatursFn = Callable[
    [
        torch.Tensor,
    ],
    torch.Tensor,
]

TMeanFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
TDriftFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
TSigmaFn = Callable[[torch.Tensor], torch.Tensor]

ProbabilityPathId = Literal[
    "constant-noise-linear-gaussian",
    "cnlg-pp",
    "schrodinger-bridge-gaussian",
    "sbg-pp",
    "variance-preserving-dirac",
    "vpd-pp",
    "linear-dirac",
    "ld-pp",
]


class ProbabilityPathDict(TypedDict):
    """"""  # noqa

    MU_T_FN_KEY: TMeanFn
    U_T_FN_KEY: TDriftFn
    SIGMA_T_FN_KEY: NotRequired[TSigmaFn]
