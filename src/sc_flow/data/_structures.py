import abc
from collections.abc import Collection, Hashable, Mapping
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any, ClassVar, Generic, TypeVar

import numpy as np
import pandas as pd

from sc_flow._types import MappedArray, NestedMappedLevelIndex, TargetCovariatesEncoderCls
from sc_flow.data._mixins import BatchMixin

__all__ = [
    "BaseData",
    "CategoricalData",
    "StateData",
    "TargetData",
    "ConditionData",
    "CompiledData",
    "NestedData",
    "NestedCompiledData",
    "MatchedData",
    "NestedMatchedData",
]


T = TypeVar("T", bound="BaseData")


@dataclass(frozen=True)
class BaseData(abc.ABC):
    """"""  # noqa

    @staticmethod
    def _get_query_idxs(
        reference_index: pd.MultiIndex,
        query_index: pd.MultiIndex,
    ) -> np.ndarray:
        return reference_index.get_indexer(query_index)

    @abc.abstractmethod
    def _slice_with_array(
        self,
        idxs: np.ndarray,
    ) -> "BaseData":
        """"""  # noqa
        raise NotImplementedError

    def slice_with_array(
        self,
        idxs: np.ndarray,
    ) -> "BaseData":
        """"""  # noqa
        return self._slice_with_array(idxs)

    def slice_with_index(
        self, reference_index: pd.MultiIndex, query_index: pd.MultiIndex, return_index_array: bool = False
    ) -> "BaseData | tuple[BaseData, np.ndarray]":
        """"""  # noqa
        idxs = self._get_query_idxs(reference_index, query_index)
        query_data = self._slice_with_array(idxs)
        if return_index_array:
            return query_data, idxs
        return query_data


@dataclass(frozen=True)
class CategoricalData(BaseData):
    """"""  # noqa

    ann_df: pd.DataFrame
    repr_dict: dict[str, MappedArray] = dc_field(default_factory=lambda: {})
    categorical_encoders: Mapping[str, TargetCovariatesEncoderCls] = dc_field(default_factory=lambda: {})

    def _slice_with_array(
        self,
        idxs: np.ndarray,
    ) -> "CategoricalData":
        """"""  # noqa
        ann_df = self.ann_df.iloc[idxs]
        return self.__class__(ann_df, self.repr_dict, self.categorical_encoders)


@dataclass(frozen=True)
class StateData(BaseData):
    """"""  # noqa

    X: np.ndarray

    def _slice_with_array(
        self,
        idxs: np.ndarray,
    ) -> "StateData":
        """"""  # noqa
        X = self.X[idxs]
        return self.__class__(X)


@dataclass(frozen=True)
class TargetData(BaseData):
    """"""  # noqa

    categorical_covariates: CategoricalData | None = None
    continuous_covariates: BatchMixin | None = None

    def _slice_with_array(
        self,
        idxs: np.ndarray,
    ) -> "TargetData":
        """"""  # noqa
        categorical_covariates = (
            None if self.categorical_covariates is None else self.categorical_covariates.slice_with_array(idxs)
        )
        continuous_covariates = (
            None if self.continuous_covariates is None else self.continuous_covariates.apply(lambda e: e[idxs])
        )
        return self.__class__(
            categorical_covariates=categorical_covariates, continuous_covariates=continuous_covariates
        )


@dataclass(frozen=True)
class ConditionData(BaseData):
    """"""  # noqa

    condition_reps: CategoricalData | None = None
    condition_covariates: BatchMixin | None = None

    def _slice_with_array(
        self,
        idxs: np.ndarray,
    ) -> "ConditionData":
        """"""  # noqa
        condition_reps = None if self.condition_reps is None else self.condition_reps.slice_with_array(idxs)
        condition_covariates = (
            None if self.condition_covariates is None else self.condition_covariates.apply(lambda e: e[idxs])
        )
        return self.__class__(condition_reps=condition_reps, condition_covariates=condition_covariates)


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

    def _slice_with_array(
        self,
        idxs: np.ndarray,
    ) -> "CompiledData":
        """"""  # noqa
        state_data = self.state_data.slice_with_array(idxs)
        target_data = None if self.target_data is None else self.target_data.slice_with_array(idxs)
        condition_data = None if self.condition_data is None else self.condition_data.slice_with_array(idxs)
        groups_data = None if self.groups_data is None else self.groups_data.slice_with_array(idxs)
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
        data: T,
        reference_index: pd.MultiIndex,
        mapped_index: NestedMappedLevelIndex,
    ) -> "NestedData":
        """"""  # noqa
        return cls._get_mapped_data_from_dict(data, reference_index, mapped_index)

    @classmethod
    def _get_leaf_mapped_data_from_dict(
        cls,
        data: T,
        reference_index: pd.MultiIndex,
        mapped_index: NestedMappedLevelIndex,
    ) -> "NestedData":
        """"""  # noqa
        if not isinstance(data, cls.dtype):
            msg = f"Leaf data is expected to be of type {cls.dtype}, found {type(data)}."
            raise TypeError(msg)
        data_dict = {
            values: data.slice_with_index(reference_index, query_index) for values, query_index in mapped_index.items()
        }
        return NestedData(data_dict)

    @classmethod
    def _get_mapped_data_from_dict(
        cls,
        data: T,
        reference_index: pd.MultiIndex,
        mapped_index: NestedMappedLevelIndex,
    ) -> "NestedData":
        if isinstance(mapped_index, NestedData):
            return cls._get_mapped_data_from_dict(
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
        index: pd.MultiIndex,
        mapped_index: dict[str, Any],
        conditions: dict[str, Collection[str]],
        control_values_dict: dict[str, str] | None,
    ) -> "MatchedData":
        """"""  # noqa
        target_data: NestedCompiledData = cls._init_target_distributions(
            data,
            index,
            mapped_index,
            conditions,
            control_values_dict,
        )
        source_data: CompiledData | None = cls._init_source_distribution(
            data,
            index,
            mapped_index,
            conditions,
            control_values_dict,
        )
        return cls(target_data, source_data=source_data)

    @classmethod
    def _init_target_distributions(
        cls,
        data: CompiledData,
        index: pd.MultiIndex,
        levels_source_value_dict: dict[str, Any],
        conditions: dict[str, Collection[str]],
        control_values_dict: dict[str, str] | None,
    ) -> NestedCompiledData:
        """"""  # noqa
        raise NotImplementedError

    @classmethod
    def _init_source_distribution(
        cls,
        data: CompiledData,
        index: pd.MultiIndex,
        levels_source_value_dict: dict[str, Any],
        conditions: dict[str, Collection[str]],
        control_values_dict: dict[str, str] | None,
    ) -> CompiledData:
        """"""  # noqa
        raise NotImplementedError


@dataclass(frozen=True)
class NestedMatchedData(NestedData):
    """"""  # noqa

    dtype: ClassVar[type] = MatchedData
