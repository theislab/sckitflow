import sys
from collections.abc import Callable

if sys.version_info < (3, 11):
    from typing_extensions import NotRequired, TypedDict
else:
    from typing import NotRequired, TypedDict

import numpy as np

try:
    from numpy.typing import NDArray

    NumpyArray = NDArray[np.float32 | np.float64]
except (ImportError, TypeError):
    NumpyArray = np.ndarray

from jax import Array as JaxArray

ArrayLike = NumpyArray | JaxArray


TMeanFn = Callable[[ArrayLike, ArrayLike, ArrayLike], ArrayLike]
TDriftFn = Callable[[ArrayLike, ArrayLike, ArrayLike, ArrayLike], ArrayLike]
TSigmaFn = Callable[[ArrayLike], ArrayLike]


class ProbabilityPathDict(TypedDict):
    """"""  # noqa

    MU_T_FN_KEY: TMeanFn
    U_T_FN_KEY: TDriftFn
    SIGMA_T_FN_KEY: NotRequired[TSigmaFn]
