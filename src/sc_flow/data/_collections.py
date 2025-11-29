from dataclasses import dataclass

import pandas as pd

from sc_flow.data._structures import CompiledData
from sc_flow.data.grouping._indexer import HierarchicalIndexer

__all__ = [
    "BaseCollection",
    "TrainCollection",
]


@dataclass
class BaseCollection:
    """"""  # noqa

    indexer: HierarchicalIndexer
    data: CompiledData

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


class TrainCollection(BaseCollection):
    """"""  # noqa
