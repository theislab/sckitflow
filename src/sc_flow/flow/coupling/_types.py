from typing import Literal, Protocol

import numpy as np
from jax import Array as JaxArray
from jax import Device
from ott.problems.linear import linear_problem

try:
    from numpy.typing import NDArray

    NumpyArray = NDArray[np.float32 | np.float64]
except (ImportError, TypeError):
    NumpyArray = np.ndarray

ArrayLike = NumpyArray | JaxArray

ScaleMethod = Literal["mean", "max", "median"] | float
LinCouplingMethod = Literal["exact", "sinkhorn", "partial", "unbalanced"]
QuadCouplingMethod = Literal["entropic_gromov_wasserstein", "entropic_fused_gromov_wasserstein"]

JaxDevice = type[Device]
DeviceLike = str | JaxDevice


class OTResult(Protocol):
    matrix: ArrayLike


class OTFn(Protocol):
    def __call__(self, problem: "linear_problem.LinearProblem") -> OTResult: ...
