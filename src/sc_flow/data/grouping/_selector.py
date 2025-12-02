from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from sc_flow._types import MappedLevelIndex, NestedMappedLevelIndex
from sc_flow.data.grouping._indexer import HierarchicalIndexer
from sc_flow.data.grouping._query import QueryFactory

__all__ = ["IndexSelector"]


@dataclass
class IndexSelector:
    """"""  # noqa

    registry: dict[str, tuple[str, ...] | None]
    hierarchy_levels: list[str]

    def __post_init__(self) -> None:
        """"""  # noqa
        self._query_factory = QueryFactory(self.registry)

    @classmethod
    def init_from_indexer(cls, indexer: HierarchicalIndexer) -> "IndexSelector":
        """"""  # noqa
        return cls(
            indexer.registry,
            indexer._hierarchy_levels,
        )

    def _get_level_query_mask(
        self,
        level_name: str,
        query_dict: dict[str, Any],
        index: pd.MultiIndex,
    ) -> np.ndarray:
        """"""  # noqa
        self._query_factory.verify_level_query_dict(level_name, query_dict)
        query_value = self._query_factory.query_dict_to_tuple(query_dict)
        level_values = index.get_level_values(level_name)
        return np.asarray(level_values == query_value)

    def _get_unique_level_values(self, level_name: str, index: pd.MultiIndex) -> Iterator[tuple[Any, ...]]:
        """"""  # noqa
        self._query_factory.verify_valid_level_name(level_name, reference=index.names)
        level_values = index.get_level_values(level_name)
        return map(tuple, level_values.unique())

    def _query_level_with_dict(
        self,
        level_name: str,
        query_dict: dict[str, Any],
        index: pd.MultiIndex,
    ) -> pd.MultiIndex:
        """"""  # noqa
        mask = self._get_level_query_mask(level_name, query_dict, index)
        return index[mask]

    def _query_level_with_tuple(
        self,
        level_name: str,
        values: tuple[Any, ...],
        index: pd.MultiIndex,
    ) -> pd.MultiIndex:
        """"""  # noqa
        query_dict = self._query_factory.query_tuple_to_dict(self.registry, values, level_name)
        return self._query_level_with_dict(level_name, query_dict, index)

    def _query_with_dict(
        self,
        query_dict: dict[str, dict[str, Any]],
        index: pd.MultiIndex,
    ) -> pd.MultiIndex:
        """"""  # noqa
        self._query_factory.verify_query_dict(query_dict)
        for level_name in self.hierarchy_levels:
            level_query_dict = query_dict[level_name]
            index = self._query_level_with_dict(
                level_name,
                level_query_dict,
                index,
            )
        return index

    def _unique_level_vals_to_dict(
        self,
        level_name: str,
        index: pd.MultiIndex,
    ) -> NestedMappedLevelIndex:
        """"""  # noqa
        unique_level_values = self._get_unique_level_values(level_name, index)
        data_dict = {values: self._query_level_with_tuple(level_name, values, index) for values in unique_level_values}
        return NestedMappedLevelIndex(data_dict)

    def _recursive_call_level_index_to_nested_dict(
        self,
        level_name: str,
        level_unique_values_dict: MappedLevelIndex,
    ) -> NestedMappedLevelIndex:
        """"""  # noqa
        data_dict = {
            values: self._level_index_to_nested_dict(
                level_name,
                values_index,
            )
            for values, values_index in level_unique_values_dict.mapping.items()
        }
        return NestedMappedLevelIndex(data_dict)

    def _level_index_to_nested_dict(
        self,
        level_name: str,
        index: pd.MultiIndex,
    ) -> NestedMappedLevelIndex:
        """"""  # noqa

        # preparing level data
        self._query_factory.verify_valid_level_name(level_name, reference=index.names)
        hierarchy_index = self.hierarchy_levels.index(level_name)
        level_unique_values_dict = self._unique_level_vals_to_dict(
            level_name,
            index,
        )
        if hierarchy_index == (self.n_hierarchy_levels - 1):
            return level_unique_values_dict
        next_level_name = self.hierarchy_levels[hierarchy_index + 1]
        return self._recursive_call_level_index_to_nested_dict(
            next_level_name,
            level_unique_values_dict,
        )

    def query_level_with_dict(
        self,
        level_name: str,
        query_dict: dict[str, Any],
        index: pd.MultiIndex,
    ) -> pd.MultiIndex:
        """"""  # noqa
        query_dict = self._query_factory.prepare_partial_level_query_dict(level_name, query_dict)
        return self._query_level_with_dict(
            level_name,
            query_dict,
            index,
        )

    def query_with_dict(
        self,
        query_dict: dict[str, dict[str, Any]],
        index: pd.MultiIndex,
    ) -> pd.MultiIndex:
        """"""  # noqa
        query_dict = self._query_factory.prepare_partial_query_dict(query_dict)
        return self._query_with_dict(query_dict, index)

    def level_index_to_nested_dict(self, level_name: str, index: pd.MultiIndex) -> NestedMappedLevelIndex:
        """"""  # noqa
        return self._level_index_to_nested_dict(level_name, index)

    def index_to_nested_dict(
        self,
        index: pd.MultiIndex,
    ) -> NestedMappedLevelIndex:
        """"""  # noqa
        return self._level_index_to_nested_dict(self.hierarchy_levels[0], index)

    @property
    def n_hierarchy_levels(self) -> int:
        """"""  # noqa
        return len(self.hierarchy_levels)

    @property
    def registry_keys(self) -> tuple[str]:
        """Returns the keys of the registry as a tuple."""
        return tuple(self.registry.keys())

    @property
    def query_factory(self) -> QueryFactory:
        """"""  # noqa
        return self._query_factory
