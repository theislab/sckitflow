from dataclasses import dataclass

import numpy as np
import pandas as pd

from sc_flow._types import MappedArray
from sc_flow.data._mixins import BatchMixin

__all__ = [
    "CategoricalData",
    "StateData",
    "TargetData",
    "ConditionData",
    "IndexedContainer",
]


@dataclass
class BaseDataContainer:
    """"""  # noqa

    pass


@dataclass
class CategoricalData(BaseDataContainer):
    """"""  # noqa

    column_values: pd.DataFrame
    repr_dict: MappedArray | None = None


@dataclass
class CombinationData(CategoricalData):
    """"""  # noqa

    combination_data: dict[str, CategoricalData]


@dataclass
class StateData(BaseDataContainer):
    """"""  # noqa

    X: np.ndarray


@dataclass
class TargetData(BaseDataContainer):
    """"""  # noqa

    categorical_covariates: CategoricalData | None = None
    continuous_covariates: BatchMixin | None = None


@dataclass
class ConditionData(BaseDataContainer):
    """"""  # noqa

    condition_reps: CombinationData | None = None
    condition_covariates: BatchMixin | None = None


@dataclass
class IndexedContainer(BaseDataContainer):
    """"""  # noqa

    state_data: StateData
    target_data: TargetData | None = None
    condition_data: ConditionData | None = None
