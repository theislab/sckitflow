from dataclasses import dataclass

import numpy as np

__all__ = ["DistributionData"]


@dataclass
class TargetData: ...


@dataclass
class ConditionData: ...


@dataclass
class DistributionData:
    X: np.ndarray
    target_data: TargetData
    condition_data: ConditionData
