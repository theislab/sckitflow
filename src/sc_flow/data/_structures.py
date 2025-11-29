from dataclasses import dataclass

import numpy as np
import pandas as pd

from sc_flow._types import MappedArray
from sc_flow.data._mixins import BatchMixin

__all__ = [
    "CategoricalDataContainer",
    "StateDataContainer",
    "TargetDataContainer",
    "ConditionDataContainer",
    "IndexedContainer",
]


@dataclass
class BaseDataContainer:
    """"""  # noqa

    pass


@dataclass
class CategoricalDataContainer(BaseDataContainer):
    """"""  # noqa

    column_values: pd.DataFrame
    repr_dict: MappedArray | None = None


@dataclass
class CombinatorialCategoricalDataContainer(CategoricalDataContainer):
    """"""  # noqa

    combination_data: dict[str, CategoricalDataContainer]


@dataclass
class StateDataContainer(BaseDataContainer):
    """"""  # noqa

    X: np.ndarray


@dataclass
class TargetDataContainer(BaseDataContainer):
    """"""  # noqa

    categorical_covariates: CategoricalDataContainer | None = None
    continuous_covariates: BatchMixin | None = None


@dataclass
class ConditionDataContainer(BaseDataContainer):
    """"""  # noqa

    condition_reps: CombinatorialCategoricalDataContainer | None = None
    condition_covariates: BatchMixin | None = None


@dataclass
class IndexedContainer(BaseDataContainer):
    """"""  # noqa

    index: pd.MultiIndex
    state_data: StateDataContainer
    target_data: TargetDataContainer | None = None
    condition_data: ConditionDataContainer | None = None
