from dataclasses import dataclass
from typing import Any

import pandas as pd

from sc_flow._constants import GROUP_LEVEL_NAME
from sc_flow.data._structures import CompiledData, MatchedData
from sc_flow.data.grouping._selector import IndexSelector

__all__ = [
    "DataCollection",
    "TrainCollection",
    "ValidationCollection",
]


class DataCollection:
    """"""  # noqa

    def __init__(self, data: CompiledData, index: pd.MultiIndex, selector: IndexSelector) -> None:
        """"""  # noqa
        self._data = data
        self._index = index
        self._selector = selector

        self._groups_data_dict: dict[tuple[Any], MatchedData] = self._get_groups_data_dict()

    @staticmethod
    def _get_unique_level_values(self, level_name: str, index: pd.MultiIndex) -> tuple[tuple[Any]]:
        """"""  # noqa
        if level_name not in self._selector.registry_keys:
            msg = f"Level {level_name} not found in registry keys {self._selector.registry_keys}"
            raise KeyError(msg)
        level_values = index.get_level_values(level_name).values
        return map(tuple, set(level_values))

    def _get_condition_data(self, group_index: pd.MultiIndex) -> None:
        """"""  # noqa
        return MatchedData
        raise NotImplementedError

    def _get_group_data(self, group_val: tuple[Any]) -> MatchedData:
        """"""  # noqa
        query_dict = self._selector.get_query_dict_from_tuple(
            group_val,
            GROUP_LEVEL_NAME,
        )
        group_index = self._selector.query_level_with_dict(GROUP_LEVEL_NAME, query_dict, self.index)
        # print
        return group_index
        # condition_data = self._get_condition_data(group_index)
        # raise NotImplementedError

    def _get_groups_data_dict(self) -> dict[tuple[Any], MatchedData]:
        """"""  # noqa
        unique_groups = self._get_unique_level_values(
            GROUP_LEVEL_NAME,
            self.index,
        )
        group_data_dict = {}
        for group_val in unique_groups:
            group_data_dict[group_val] = self._get_group_data(group_val)
        return group_data_dict

    @property
    def groups_data_dict(self) -> dict[tuple[Any], MatchedData]:
        """"""  # noqa
        return self._groups_data_dict

    @property
    def data(self) -> CompiledData:
        """"""  # noqa
        return self._data

    @property
    def index(self) -> pd.MultiIndex:
        """"""  # noqa
        return self._index

    @property
    def selector(self) -> IndexSelector:
        """"""  # noqa
        return self._selector


@dataclass
class TrainCollection(DataCollection):
    """"""  # noqa

    def __init__(self, data: CompiledData, index: pd.MultiIndex, selector: IndexSelector) -> None:
        """"""  # noqa
        super().__init__(data, index, selector)


@dataclass
class ValidationCollection(DataCollection):
    """"""  # noqa

    def __init__(self, data: CompiledData, index: pd.MultiIndex, selector: IndexSelector) -> None:
        """"""  # noqa
        super().__init__(data, index, selector)
