from collections.abc import Callable, Collection
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

import jax.numpy as jnp
from jax import Array as JaxArray
from jax import Device

ArrayLike = NumpyArray | JaxArray

MappedArray = dict[str, ArrayLike]

TTimeFeaturesFn = Callable[[ArrayLike, int], ArrayLike]

TMeanFn = Callable[[ArrayLike, ArrayLike, ArrayLike], ArrayLike]
TDriftFn = Callable[[ArrayLike, ArrayLike, ArrayLike, ArrayLike], ArrayLike]
TSigmaFn = Callable[[ArrayLike], ArrayLike]
ScaleMethod = Literal["mean", "max", "median"] | float
LinCouplingMethod = Literal["exact", "sinkhorn", "partial", "unbalanced"]
QuadCouplingMethod = Literal["entropic_gromov_wasserstein", "entropic_fused_gromov_wasserstein"]

TVfFn = Callable[[ArrayLike, ArrayLike], ArrayLike]

MatchFnOut = tuple[ArrayLike, ArrayLike] | tuple[ArrayLike, ArrayLike, ArrayLike]
TMatchFn = Callable[[ArrayLike, ArrayLike], MatchFnOut]

TTimeSamplerFn = Callable[[tuple[int, ...]], ArrayLike] | Callable[[tuple[int, ...]], tuple[ArrayLike, ArrayLike]]
TNoiseSamplerFn = Callable[[tuple[int, ...]], ArrayLike]


class TConditioningFn(Protocol):
    def __call__(
        self,
        encoded_t: ArrayLike,
        encoded_state: ArrayLike,
        *args: ArrayLike,
    ) -> ArrayLike: ...


JaxDevice = type[Device]
TDevice = str | JaxDevice

TTimeStateDiffusion = Callable[[ArrayLike, ArrayLike, Any], ArrayLike]
TTimeDiffusion = Callable[[ArrayLike, Any], ArrayLike]
TDiffusion = TTimeDiffusion | TTimeStateDiffusion


TODEDynamics = TypeVar("TODEDynamics", bound="BaseVelocityField")
TSDEDynamics = tuple[TODEDynamics, TDiffusion]
TSolverDynamics = TypeVar("TSolverDynamics", TODEDynamics, TSDEDynamics)


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

    @classmethod
    def concatenate(cls, preds: Collection["PredictionData"], axis: int = 0) -> "PredictionData":
        samples = jnp.concatenate([p.samples for p in preds], axis=axis)
        trajs = [p.traj for p in preds if p.traj is not None]
        traj = jnp.concatenate(trajs, axis=axis) if trajs else None
        return cls(samples=samples, traj=traj)
