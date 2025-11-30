from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field as dc_field

import numpy as np
import pandas as pd

from sc_flow._types import MappedArray, TargetCovariatesEncoderCls
from sc_flow.data._mixins import BatchMixin

__all__ = [
    "CategoricalData",
    "CombinationData",
    "StateData",
    "TargetData",
    "ConditionData",
    "CompiledData",
]


@dataclass(frozen=True)
class BaseData:
    """"""  # noqa

    pass


@dataclass(frozen=True)
class CategoricalData(BaseData):
    """"""  # noqa

    ann_df: pd.DataFrame
    repr_dict: dict[str, MappedArray] = dc_field(default_factory=lambda: {})
    categorical_encoders: Mapping[str, TargetCovariatesEncoderCls] = dc_field(default_factory=lambda: {})


@dataclass(frozen=True)
class CombinationData(CategoricalData):
    """"""  # noqa


@dataclass(frozen=True)
class StateData(BaseData):
    """"""  # noqa

    X: np.ndarray


@dataclass(frozen=True)
class TargetData(BaseData):
    """"""  # noqa

    categorical_covariates: CategoricalData | None = None
    continuous_covariates: BatchMixin | None = None


@dataclass(frozen=True)
class ConditionData(BaseData):
    """"""  # noqa

    condition_reps: CombinationData | None = None
    condition_covariates: BatchMixin | None = None


@dataclass(frozen=True)
class CompiledData(BaseData):
    """"""  # noqa

    state_data: StateData
    target_data: TargetData | None = None
    condition_data: ConditionData | None = None
    groups_data: CategoricalData | None = None

    @property
    def ann_df(self) -> pd.DataFrame:
        """"""  # noqa
        return self.condition_data.condition_reps.ann_df
