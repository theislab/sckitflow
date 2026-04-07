from collections.abc import Sequence

import jax
import jax.numpy as jnp

from sc_flow.backends.jax._types import ArrayLike
from sc_flow.backends.jax.metrics.__types import EnergyDistanceState, MaximumMeanDiscrepancyState
from sc_flow.backends.jax.metrics._utils import compute_e_distance_fast


@jax.jit
def rbf_kernel_fast(x: ArrayLike, y: ArrayLike, gamma: float) -> ArrayLike:
    xx = (x**2).sum(1)
    yy = (y**2).sum(1)
    xy = x @ y.T
    sq_distances = xx[:, None] + yy - 2 * xy
    return jnp.exp(-gamma * sq_distances)


class EnergyDistance:
    def init(self) -> EnergyDistanceState:
        return EnergyDistanceState(
            energy_distance_raw=jnp.zeros((), dtype=jnp.float32), total=jnp.zeros((), dtype=jnp.float32)
        )

    @jax.jit
    def update(self, state: EnergyDistanceState, pred: ArrayLike, target: ArrayLike) -> EnergyDistanceState:
        e_dist = compute_e_distance_fast(pred, target)
        return EnergyDistanceState(energy_distance_raw=state.energy_distance_raw + e_dist, total=state.total + 1)

    def compute(self, state: EnergyDistanceState) -> float:
        return state.energy_distance_raw / state.total

    def reset(self) -> EnergyDistanceState:
        return self.__init__()


class MaximumMeanDiscrepancy:
    def __init__(self, gammas: Sequence[float] | None = None):
        if gammas is None:
            gammas = [1.0]
        self.gammas = list(gammas)

    def _compute_mmd(self, pred: ArrayLike, target: ArrayLike) -> float:
        mmds = []
        for gamma in self.gammas:
            xx = rbf_kernel_fast(pred, pred, gamma)
            yy = rbf_kernel_fast(target, target, gamma)
            xy = rbf_kernel_fast(pred, target, gamma)
            mmd = jnp.mean(xx) + jnp.mean(yy) - 2 * jnp.mean(xy)
            mmds.append(mmd)
        return jnp.mean(jnp.array(mmds))

    def update(
        self, state: MaximumMeanDiscrepancyState, pred: ArrayLike, target: ArrayLike
    ) -> MaximumMeanDiscrepancyState:
        mmd = self._compute_mmd(pred, target)
        return MaximumMeanDiscrepancyState(mmd_raw=state.mmd_raw + mmd, total=state.total + 1)

    def compute(self, state: MaximumMeanDiscrepancyState) -> float:
        return state.mmd_raw / state.total

    def reset(self) -> MaximumMeanDiscrepancyState:
        return self.init()
