from collections.abc import Callable
from typing import Protocol

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

class TConditioningFn(Protocol):
    def __call__(
        self,
        encoded_t: ArrayLike,
        encoded_state: ArrayLike,
        *args: ArrayLike,
    ) -> ArrayLike: ...