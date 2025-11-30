from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

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
    ) -> pd.Series | np.ndarray:
        """"""  # noqa
        self._query_factory.verify_level_query_dict(level_name, query_dict)
        query_value = self._query_factory.query_dict_to_tuple(query_dict)
        level_values = index.get_level_values(level_name)
        return level_values == query_value

    def _get_unique_level_values(self, level_name: str, index: pd.MultiIndex) -> tuple[tuple[Any]]:
        """"""  # noqa
        self._query_factory.verify_valid_level_name(level_name, reference=index.names)
        level_values = index.get_level_values(level_name).values
        return map(tuple, set(level_values))

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

    def _unique_level_vals_to_dict(self, level_name: str, index: pd.MultiIndex) -> tuple[tuple[Any]]:
        """"""  # noqa
        unique_level_values = self._get_unique_level_values(level_name, index)
        return {values: self._query_level_with_tuple(level_name, values, index) for values in unique_level_values}

    def _level_index_to_nested_dict(
        self,
        level_name: str,
        index: pd.MultiIndex,
    ) -> dict:
        """"""  # noqa

        # preparing level data
        self._query_factory.verify_valid_level_name(level_name, reference=index.names)
        hierarchy_index = self.hierarchy_levels.index(level_name)
        current_level_unique_values_dict = self._unique_level_vals_to_dict(level_name, index)
        # last level (create dict)
        if hierarchy_index == (self.n_hierarchy_levels - 1):
            return current_level_unique_values_dict
        # intermediate level (recursive call on each leaf)
        else:
            next_level_name = self.hierarchy_levels[hierarchy_index + 1]
            return {
                values: self._level_index_to_nested_dict(
                    next_level_name,
                    values_index,
                )
                for values, values_index in current_level_unique_values_dict.items()
            }

    def query_level_with_dict(
        self,
        level_name: str,
        query_dict: dict[str, Any],
        index: pd.MultiIndex,
    ) -> pd.MultiIndex:
        """"""  # noqa
        query_dict = self._prepare_partial_level_query_dict(level_name, query_dict)
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
        query_dict = self._prepare_partial_query_dict(query_dict)
        return self._query_with_dict(query_dict, index)

    def level_index_to_nested_dict(
        self,
        level_name: str,
        index: pd.MultiIndex,
    ) -> None:
        """"""  # noqa
        return self._level_index_to_nested_dict(level_name, index)

    def index_to_nested_dict(
        self,
        index: pd.MultiIndex,
    ) -> None:
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
