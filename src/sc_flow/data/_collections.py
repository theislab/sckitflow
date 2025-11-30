from dataclasses import dataclass
from typing import Any

import pandas as pd

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

        self._groups_data_dict: dict[tuple[Any], MatchedData] = self._selector.index_to_nested_dict(self.index)

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
