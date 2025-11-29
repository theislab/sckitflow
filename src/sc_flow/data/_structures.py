from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sc_flow._types import MappedArray, TargetCovariatesEncoderCls
from sc_flow.data._mixins import BatchMixin

__all__ = [
    "CategoricalData",
    "StateData",
    "TargetData",
    "ConditionData",
    "CompiledData",
]


@dataclass
class BaseData:
    """"""  # noqa

    pass


@dataclass
class CategoricalData(BaseData):
    """"""  # noqa

    ann_df: pd.DataFrame
    repr_dict: MappedArray | None = None
    categorical_encoders: Mapping[str, TargetCovariatesEncoderCls] | None = None


@dataclass
class GroupsData(BaseData):
    """"""  # noqa

    combination_data: dict[str, CategoricalData]


@dataclass
class CombinationData(GroupsData):
    """"""  # noqa


@dataclass
class StateData(BaseData):
    """"""  # noqa

    X: np.ndarray


@dataclass
class TargetData(BaseData):
    """"""  # noqa

    categorical_covariates: CategoricalData | None = None
    continuous_covariates: BatchMixin | None = None


@dataclass
class ConditionData(BaseData):
    """"""  # noqa

    condition_reps: CombinationData | None = None
    condition_covariates: BatchMixin | None = None


@dataclass
class CompiledData(BaseData):
    """"""  # noqa

    state_data: StateData
    target_data: TargetData | None = None
    condition_data: ConditionData | None = None
    groups_data: GroupsData | None = None

    @property
    def ann_df(self) -> pd.DataFrame:
        """"""  # noqa
        return self.condition_data.condition_reps.ann_df
