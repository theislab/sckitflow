from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from sc_flow._types import MappedArray, MappedCovariatesEncoder
from sc_flow.data._mixins import BatchMixin

__all__ = [
    "CategoricalData",
    "StateData",
    "TargetData",
    "ConditionData",
    "CompiledData",
]


@dataclass
class BaseDataContainer:
    """"""  # noqa

    pass


@dataclass
class CategoricalData(BaseDataContainer):
    """"""  # noqa

    ann_df: pd.DataFrame
    repr_dict: MappedArray | None = None
    categorical_encoders: MappedCovariatesEncoder | None = None


@dataclass
class CombinationData(CategoricalData):
    """"""  # noqa

    combination_data: dict[str, Any]


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
class CompiledData(BaseDataContainer):
    """"""  # noqa

    state_data: StateData
    target_data: TargetData | None = None
    condition_data: ConditionData | None = None

    @property
    def ann_df(self) -> pd.DataFrame:
        """"""  # noqa
        return self.condition_data.condition_reps.ann_df
