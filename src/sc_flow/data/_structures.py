import abc
from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import TypeVar

import numpy as np
import pandas as pd

from sc_flow._types import MappedArray, NestedMappedLevelIndex, TargetCovariatesEncoderCls
from sc_flow.data._mixins import BatchMixin

__all__ = [
    "CategoricalData",
    "CombinationData",
    "StateData",
    "TargetData",
    "ConditionData",
    "CompiledData",
]


T = TypeVar("T", "NestedData", "NestedCompiledData", "NestedMatchedData")


@dataclass(frozen=True)
class BaseData(abc.ABC):
    """"""  # noqa

    @abc.abstractmethod
    def slice_with_index(
        self,
        index: pd.MultiIndex,
    ) -> "BaseData":
        """"""  # noqa
        raise NotImplementedError


@dataclass(frozen=True)
class CategoricalData(BaseData):
    """"""  # noqa

    ann_df: pd.DataFrame
    repr_dict: dict[str, MappedArray] = dc_field(default_factory=lambda: {})
    categorical_encoders: Mapping[str, TargetCovariatesEncoderCls] = dc_field(default_factory=lambda: {})

    def slice_with_index(
        self,
        index: pd.MultiIndex,
    ) -> "CategoricalData":
        """"""  # noqa
        raise NotImplementedError


@dataclass(frozen=True)
class CombinationData(CategoricalData):
    """"""  # noqa


@dataclass(frozen=True)
class StateData(BaseData):
    """"""  # noqa

    X: np.ndarray

    def slice_with_index(
        self,
        index: pd.MultiIndex,
    ) -> "StateData":
        """"""  # noqa
        raise NotImplementedError


@dataclass(frozen=True)
class TargetData(BaseData):
    """"""  # noqa

    categorical_covariates: CategoricalData | None = None
    continuous_covariates: BatchMixin | None = None

    def slice_with_index(
        self,
        index: pd.MultiIndex,
    ) -> "TargetData":
        """"""  # noqa
        raise NotImplementedError


@dataclass(frozen=True)
class ConditionData(BaseData):
    """"""  # noqa

    condition_reps: CombinationData | None = None
    condition_covariates: BatchMixin | None = None

    def slice_with_index(
        self,
        index: pd.MultiIndex,
    ) -> "ConditionData":
        """"""  # noqa
        raise NotImplementedError


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
        conditions_df = self.condition_data.condition_reps.ann_df
        groups_df = self.groups_data.ann_df
        return pd.concat((conditions_df, groups_df), axis=1)

    def slice_with_index(
        self,
        index: pd.MultiIndex,
    ) -> "CompiledData":
        """"""  # noqa
        raise NotImplementedError


@dataclass(frozen=True)
class NestedData(BaseData, abc.ABC):
    """"""  # noqa

    @classmethod
    def init_from_nested_index(cls, data: BaseData, mapped_index: NestedMappedLevelIndex) -> "NestedData":
        """"""  # noqa
        raise NotImplementedError


@dataclass(frozen=True)
class NestedCompiledData(NestedData, CompiledData):
    """"""  # noqa

    data: Mapping[Hashable, CompiledData | T]
    mapped_index: NestedMappedLevelIndex

    @classmethod
    def init_from_nested_index_dict(
        cls, data: CompiledData, mapped_index: NestedMappedLevelIndex
    ) -> "NestedCompiledData":
        """"""  # noqa
        return cls._init_from_nested_index_dict(
            data,
            mapped_index,
        )

    @classmethod
    def _init_from_nested_index_dict(data: CompiledData, mapped_index: NestedMappedLevelIndex) -> "NestedCompiledData":
        """"""  # noqa


@dataclass(frozen=True)
class MatchedData(BaseData):
    """"""  # noqa

    index: pd.MultiIndex
    target_data: NestedCompiledData
    source_data: CompiledData | None = None


@dataclass(frozen=True)
class NestedMatchedData(NestedData, MatchedData):
    """"""  # noqa
