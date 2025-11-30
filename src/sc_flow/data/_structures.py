import abc
from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any, ClassVar, Generic, TypeVar

import numpy as np
import pandas as pd

from sc_flow._types import MappedArray, NestedMappedLevelIndex, TargetCovariatesEncoderCls
from sc_flow.data._mixins import BatchMixin

__all__ = [
    "CategoricalData",
    "StateData",
    "TargetData",
    "ConditionData",
    "CompiledData",
]


T = TypeVar("T", bound="BaseData")


@dataclass(frozen=True)
class BaseData(abc.ABC):
    """"""  # noqa

    @abc.abstractmethod
    def slice_with_index(
        self,
        reference_index: pd.MultiIndex,
        query_index: pd.MultiIndex,
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
        reference_index: pd.MultiIndex,
        query_index: pd.MultiIndex,
    ) -> "CategoricalData":
        """"""  # noqa
        raise NotImplementedError


@dataclass(frozen=True)
class StateData(BaseData):
    """"""  # noqa

    X: np.ndarray

    def slice_with_index(
        self,
        reference_index: pd.MultiIndex,
        query_index: pd.MultiIndex,
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
        reference_index: pd.MultiIndex,
        query_index: pd.MultiIndex,
    ) -> "TargetData":
        """"""  # noqa
        raise NotImplementedError


@dataclass(frozen=True)
class ConditionData(BaseData):
    """"""  # noqa

    condition_reps: CategoricalData | None = None
    condition_covariates: BatchMixin | None = None

    def slice_with_index(
        self,
        reference_index: pd.MultiIndex,
        query_index: pd.MultiIndex,
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
        reference_index: pd.MultiIndex,
        query_index: pd.MultiIndex,
    ) -> "CompiledData":
        """"""  # noqa
        state_data = self.state_data.slice_with_index(reference_index, query_index)
        target_data = self.target_data.slice_with_index(reference_index, query_index)
        condition_data = self.condition_data.slice_with_index(reference_index, query_index)
        groups_data = self.groups_data.slice_with_index(reference_index, query_index)
        return self.__class__(
            state_data, target_data=target_data, condition_data=condition_data, groups_data=groups_data
        )


@dataclass(frozen=True)
class NestedData(Generic[T]):
    """"""  # noqa

    dtype: ClassVar[type] = BaseData
    data: Mapping[Hashable, "T | NestedData[T]"]

    @classmethod
    def init_from_data(
        cls,
        data: BaseData,
        reference_index: pd.MultiIndex,
        mapped_index: NestedMappedLevelIndex,
    ) -> "NestedData":
        """"""  # noqa
        return cls._get_mapped_data_from_dict(data, reference_index, mapped_index)

    @classmethod
    def _get_leaf_mapped_data_from_dict(
        cls,
        data: BaseData,
        reference_index: pd.MultiIndex,
        mapped_index: NestedMappedLevelIndex,
    ) -> "NestedData":
        """"""  # noqa
        if not isinstance(data, cls.dtype):
            msg = f"Leaf data is expected to be of type {cls.dtype}, found {type(data)}."
            raise TypeError(msg)
        data_dict = {
            values: data.slice_with_index(query_index, reference_index) for values, query_index in mapped_index.items()
        }
        return NestedData(data_dict)

    @classmethod
    def _get_mapped_data_from_dict(
        cls,
        data: BaseData,
        reference_index: pd.MultiIndex,
        mapped_index: NestedMappedLevelIndex,
    ) -> "NestedData":
        if isinstance(mapped_index, NestedData):
            return cls._get_leaf_mapped_data_from_dict(
                data,
                reference_index,
                mapped_index,
            )
        else:
            return cls._get_leaf_mapped_data_from_dict(data, reference_index, mapped_index)


@dataclass(frozen=True)
class NestedCompiledData(NestedData):
    """"""  # noqa

    dtype: ClassVar[type] = CompiledData


@dataclass(frozen=True)
class MatchedData:
    """"""  # noqa

    target_data: NestedCompiledData
    source_data: CompiledData | None = None

    @classmethod
    def init_from_data(
        cls,
        data: CompiledData,
        levels_source_value_dict: dict[str, Any],
    ) -> "MatchedData":
        """"""  # noqa
        target_data: NestedCompiledData = cls._init_target_distributions(
            data,
            levels_source_value_dict,
        )
        source_data: CompiledData | None = cls._init_source_distribution(
            data,
            levels_source_value_dict,
        )
        return cls(target_data, source_data=source_data)

    @classmethod
    def _init_target_distributions(
        cls,
        data: CompiledData,
        levels_source_value_dict: dict[str, Any],
    ) -> NestedCompiledData:
        """"""  # noqa
        raise NotImplementedError

    @classmethod
    def _init_source_distribution(
        cls,
        data: CompiledData,
        levels_source_value_dict: dict[str, Any],
    ) -> CompiledData:
        """"""  # noqa
        raise NotImplementedError


@dataclass(frozen=True)
class NestedMatchedData(NestedData):
    """"""  # noqa

    dtype: ClassVar[type] = MatchedData
