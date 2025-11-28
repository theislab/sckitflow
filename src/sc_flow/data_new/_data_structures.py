from dataclasses import dataclass

import numpy as np

__all__ = ["DistributionData"]


@dataclass
class StateData:
    X: np.ndarray


@dataclass
class TargetData: ...


@dataclass
class ConditionData: ...


@dataclass
class DistributionData:
    state_data: StateData
    target_data: TargetData
    condition_data: ConditionData


@dataclass
class MatchedDistributionsData:
    target_distr: DistributionData
    source_distr: DistributionData | None = None
