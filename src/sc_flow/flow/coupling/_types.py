from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, Protocol, TypeVar

import diffrax as dfx
import numpy as np
from ott.problems.linear import linear_problem

if TYPE_CHECKING:
    from sc_flow.backends.jax.nn._vf import BaseVelocityField
try:
    from numpy.typing import NDArray

    NumpyArray = NDArray[np.float32 | np.float64]
except (ImportError, TypeError):
    NumpyArray = np.ndarray

from jax import Array as JaxArray
from jax import Device

ArrayLike = NumpyArray | JaxArray

MappedArray = dict[str, ArrayLike]

TimeFeaturesFn = Callable[[ArrayLike, int], ArrayLike]

MeanFn = Callable[[ArrayLike, ArrayLike, ArrayLike], ArrayLike]
DriftFn = Callable[[ArrayLike, ArrayLike, ArrayLike, ArrayLike], ArrayLike]
SigmaFn = Callable[[ArrayLike], ArrayLike]
ScaleMethod = Literal["mean", "max", "median"] | float
LinCouplingMethod = Literal["exact", "sinkhorn", "partial", "unbalanced"]
QuadCouplingMethod = Literal["entropic_gromov_wasserstein", "entropic_fused_gromov_wasserstein"]

VelocityFieldFn = Callable[[ArrayLike, ArrayLike], ArrayLike]

MatchFnOut = tuple[ArrayLike, ArrayLike] | tuple[ArrayLike, ArrayLike, ArrayLike]
MatchFn = Callable[[ArrayLike, ArrayLike], MatchFnOut]

TimeSamplerFn = Callable[[tuple[int, ...]], ArrayLike] | Callable[[tuple[int, ...]], tuple[ArrayLike, ArrayLike]]
NoiseSamplerFn = Callable[[tuple[int, ...]], ArrayLike]


class CombinerFn(Protocol):
    def __call__(
        self,
        encoded_t: ArrayLike,
        encoded_state: ArrayLike,
        *args: ArrayLike,
    ) -> ArrayLike: ...


JaxDevice = type[Device]
DeviceLike = str | JaxDevice

TimeStateDiffusion = Callable[[ArrayLike, ArrayLike, Any], ArrayLike]
TimeDiffusion = Callable[[ArrayLike, Any], ArrayLike]
Diffusion = TimeDiffusion | TimeStateDiffusion


ODEDynamics = TypeVar("ODEDynamics", bound="BaseVelocityField")
SDEDynamics = tuple[ODEDynamics, Diffusion]
SolverDynamics = TypeVar("SolverDynamics", ODEDynamics, SDEDynamics)


class SolverConfig(NamedTuple):
    """Configuration extracted from solver_kwargs and common parameters."""

    dt0: float
    max_steps: int
    stepsize_controller: dfx.AbstractStepSizeController
    saveat: dfx.SaveAt
    source_on_device: ArrayLike
    remaining_kwargs: dict[str, Any]


class OTResult(Protocol):
    matrix: ArrayLike


class OTFn(Protocol):
    def __call__(self, problem: "linear_problem.LinearProblem") -> OTResult: ...


@dataclass
class PredictionData:
    samples: ArrayLike
    traj: ArrayLike | None = None
