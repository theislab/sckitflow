from collections.abc import Collection
from dataclasses import dataclass
from typing import ClassVar, Generic, TypeVar

import pandas as pd

from sc_flow._types import NestedMappedLevelIndex
from sc_flow.data._mixins import DataMixin
from sc_flow.data._structures import BaseData, CategoricalData, CouplingData, DistributionData, MixedTypeData, StateData

__all__ = [
    "NestedData",
    "MatchedData",
]

T = TypeVar("T", "BaseData", "CategoricalData", "StateData", "MixedTypeData", "CouplingData", "DistributionData")


@dataclass(frozen=True)
class NestedData(DataMixin, Generic[T]):
    """"""  # noqa

    strict: ClassVar[bool] = False
    required_value_type: ClassVar[type] = BaseData

    @classmethod
    def init_from_data(
        cls,
        data: T,
        reference_index: pd.MultiIndex,
        mapped_index: NestedMappedLevelIndex,
        conditions: dict[str, Collection[str]] | None,
        control_values_dict: dict[str, str] | None,
    ) -> "NestedData":
        """"""  # noqa
        return cls._get_mapped_data_from_dict(data, reference_index, mapped_index, conditions, control_values_dict)

    @classmethod
    def _get_leaf_mapped_data_from_dict(
        cls,
        data: T,
        reference_index: pd.MultiIndex,
        mapped_index: NestedMappedLevelIndex,
        conditions: dict[str, Collection[str]] | None,
        control_values_dict: dict[str, str] | None,
    ) -> T:
        if not isinstance(data, cls.required_value_type):
            msg = f"Leaf data is expected to be of type {cls.dtype}, found {type(data)}."
            raise TypeError(msg)
        data = data.slice_with_index(reference_index, mapped_index)
        return data

    @classmethod
    def _get_mapped_data_from_dict(
        cls,
        data: T,
        reference_index: pd.MultiIndex,
        mapped_index: NestedMappedLevelIndex,
        conditions: dict[str, Collection[str]] | None,
        control_values_dict: dict[str, str] | None,
    ) -> "NestedData":
        out_dict = {}
        for key, value in mapped_index.mapping.items():
            if isinstance(value, NestedMappedLevelIndex):
                out_dict[key] = cls._get_mapped_data_from_dict(
                    data, reference_index, value, conditions, control_values_dict
                )
            else:
                out_dict[key] = cls._get_leaf_mapped_data_from_dict(
                    data, reference_index, value, conditions, control_values_dict
                )
        return cls(out_dict)

    def _slice_with_array(self, idxs):
        return self._apply(self.mapping, lambda e: e.slice_with_array(idxs))


@dataclass(frozen=True)
class MatchedData:
    """"""  # noqa

    target_data: NestedData[DistributionData]
    source_data: DistributionData | None = None
    coupling_data: NestedData[CouplingData] | None = None
