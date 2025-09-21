from collections.abc import Callable
from typing import Union

import numpy as np

try:
    from numpy.typing import NDArray

    NumpyArray = NDArray[np.float32 | np.float64]
except (ImportError, TypeError):
    NumpyArray = np.ndarray

from jax import Array as JaxArray

ArrayLike = Union[NDArray, JaxArray]

TTimeFeaturesFn = Callable[[ArrayLike, int], ArrayLike]

TMeanFn = Callable[[ArrayLike, ArrayLike, ArrayLike], ArrayLike]
TDriftFn = Callable[[ArrayLike, ArrayLike, ArrayLike, ArrayLike], ArrayLike]
TSigmaFn = Callable[[ArrayLike], ArrayLike]
