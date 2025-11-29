from dataclasses import dataclass

import pandas as pd

from sc_flow.data._structures import CompiledData
from sc_flow.data.grouping._indexer import HierarchicalIndexer
from sc_flow.data.grouping._selector import IndexSelector

__all__ = [
    "BaseCollection",
    "DataCollection",
    "TrainCollection",
    "ValidationCollection",
]


@dataclass
class BaseCollection:
    """"""  # noqa

    data: CompiledData
    indexer: HierarchicalIndexer
    selector: IndexSelector

    def __post_init__(self) -> None:
        """"""  # noqa
        self._index = self._get_index()

    def _get_index(self) -> pd.MultiIndex:
        """"""  # noqa
        return self.indexer.create_index(self.data.ann_df)

    @property
    def index(self) -> pd.MultiIndex:
        """"""  # noqa
        return self._index


class DataCollection(BaseCollection):
    """"""  # noqa

    def __post_init__(self):
        super().__post_init__()
        self._prepare_groups()

    def _prepare_groups(self) -> None:
        """"""  # noqa


class TrainCollection(DataCollection):
    """"""  # noqa


class ValidationCollection(DataCollection):
    """"""  # noqa
