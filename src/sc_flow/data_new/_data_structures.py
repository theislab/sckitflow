from dataclasses import dataclass

import numpy as np

from sc_flow.data_new._mixins import ArrayMixin, BatchMixin

__all__ = ["DistributionData"]


@dataclass
class StateData:
    X: np.ndarray


@dataclass
class TargetData:
    target_data: BatchMixin


@dataclass
class ConditionData:
    condition_reps: dict[str, ArrayMixin]
    condition_covariates: BatchMixin


@dataclass
class DistributionData:
    state_data: StateData
    target_data: TargetData
    condition_data: ConditionData


@dataclass
class MatchedDistributionsData:
    target_distr: DistributionData
    source_distr: DistributionData | None = None
