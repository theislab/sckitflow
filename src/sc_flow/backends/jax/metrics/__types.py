from typing import NamedTuple

from sc_flow.backends.jax._types import ArrayLike


class EnergyDistanceState(NamedTuple):
    energy_distance_raw: ArrayLike
    total: ArrayLike


class MaximumMeanDiscrepancyState(NamedTuple):
    mmd_raw: ArrayLike
    total: ArrayLike


class SinkhornDivergenceState(NamedTuple):
    sinkhorn_raw: ArrayLike
    total: ArrayLike


class ClassificationState(NamedTuple):
    preds: list
    targets: list
