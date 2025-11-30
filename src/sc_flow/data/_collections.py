from dataclasses import dataclass

import pandas as pd

from sc_flow.data._structures import CompiledData, NestedCompiledData, NestedMatchedData
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

        self._matched_data: NestedMatchedData = self._get_groups_data_dict()

    def _get_groups_data_dict(self) -> NestedMatchedData:
        """"""  # noqa
        mapped_index = self._selector.index_to_nested_dict(self.index)
        # return NestedMatchedData.init_from_data(self._data, self.index, mapped_index)
        return NestedCompiledData.init_from_data(self._data, self.index, mapped_index)

    @property
    def data(self) -> CompiledData:
        """"""  # noqa
        return self._data

    @property
    def matched_data(self) -> NestedMatchedData:
        """"""  # noqa
        return self._matched_data

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
