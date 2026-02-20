from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from sc_flow.data._mixins import MappedLevelIndex
from sc_flow.data.grouping._indexer import HierarchicalIndexer
from sc_flow.data.grouping._query import QueryFactory

__all__ = ["IndexSelector"]


@dataclass
class IndexSelector:
    """Class to handle selection from a hierarchical indexer."""

    def __init__(
        self,
        registry: dict[str, tuple[str, ...] | None],
        hierarchy_levels: list[str],
    ) -> None:
        """Initializes the selector.

        :param registry: The dictionary containing the registry for the columns
            corresponding to each level.
        :type registry: class: `dict[str, tuple[str, ...] | None]`

        :param hierarchy_levels: The list indicating the hierarchy of each level.
            The hierarchy is resolved sequentially along this list in decreasing order.
            Hence, levels appearing earlier in the list will have higher hierarchy order.
        :type hierarchy_levels: class: `hierarchy_levels: list[str]`
        """
        self._registry = registry
        self._hierarchy_levels = hierarchy_levels

        self._query_factory = QueryFactory(self._registry)

    @classmethod
    def init_from_indexer(cls, indexer: HierarchicalIndexer) -> "IndexSelector":
        """Initializes the selector from an indexer. Its registry and hierarchy levels will be used.

        :param indexer: The indexer used to initialize the selector.
        :class indexer: class: `HierarchicalIndexer`
        """
        return cls(
            indexer.registry,
            indexer.hierarchy_levels,
        )

    def _get_sublevel_positions(
        self, level_name: str, reference_index: pd.MultiIndex,
    ) -> list[int]:
        """Returns the positional indices of sub-levels belonging to a hierarchy level."""
        return [
            i for i, name in enumerate(reference_index.names)
            if isinstance(name, tuple) and name[0] == level_name
        ]

    def _get_unique_level_values(self, level_name: str, reference_index: pd.MultiIndex) -> Iterator[tuple[Any, ...]]:
        """Returns the unique values for a given level from a reference index.

        :param level_name: String identifier for the level to query.
        :type level_name: class: `str`

        :param reference_index: The index which to retrieve the unique values of.
        :type reference_index: class: `pd.MultiIndex`
        """
        self._query_factory.verify_valid_level_name(level_name)
        positions = self._get_sublevel_positions(level_name, reference_index)
        if len(positions) == 1:
            categories = reference_index.levels[positions[0]]
            codes = np.asarray(reference_index.codes[positions[0]])
            present = np.unique(codes)
            return ((categories[c],) for c in present)
        col_uniques = []
        for p in positions:
            cats = reference_index.levels[p]
            present = np.unique(np.asarray(reference_index.codes[p]))
            col_uniques.append([cats[c] for c in present])
        from itertools import product
        return (combo for combo in product(*col_uniques))

    def _query_level_with_dict(
        self,
        level_name: str,
        query_dict: dict[str, Any],
        reference_index: pd.MultiIndex,
    ) -> pd.MultiIndex:
        """Queries a reference index on a given level with the input query dictionary.

        Uses searchsorted on integer codes since data is sorted by hierarchy
        levels. Falls back to code-based masking if values are not contiguous.

        :param level_name: String identifier for the level to query.
        :type level_name: class: `str`

        :param query_dict: The dictionary used to construct the query.
        :type query_dict: class: `dict[str, Any]`

        :param reference_index: The index which to query.
        :type reference_index: class: `pd.MultiIndex`
        """
        self._query_factory.verify_level_query_dict(level_name, query_dict)
        active = {col: val for col, val in query_dict.items() if val != slice(None)}
        if not active:
            return reference_index

        if len(active) == 1:
            col, val = next(iter(active.items()))
            pos = reference_index.names.index((level_name, col))
            codes = np.asarray(reference_index.codes[pos])
            categories = reference_index.levels[pos]
            target_code = categories.get_loc(val)
            left = np.searchsorted(codes, target_code, side="left")
            right = np.searchsorted(codes, target_code, side="right")
            return reference_index[left:right]

        # Multi-column: intersect ranges from each column
        left, right = 0, len(reference_index)
        for col, val in active.items():
            pos = reference_index.names.index((level_name, col))
            codes = np.asarray(reference_index.codes[pos])
            categories = reference_index.levels[pos]
            target_code = categories.get_loc(val)
            sub_codes = codes[left:right]
            sub_left = np.searchsorted(sub_codes, target_code, side="left")
            sub_right = np.searchsorted(sub_codes, target_code, side="right")
            left = left + sub_left
            right = left + (sub_right - sub_left)
        return reference_index[left:right]

    def _query_level_with_tuple(
        self,
        level_name: str,
        values: tuple[Any, ...],
        reference_index: pd.MultiIndex,
    ) -> pd.MultiIndex:
        """Queries a reference index on a given level with the input query tuple.

        :param level_name: String identifier for the level to query.
        :type level_name: class: `str`

        :param values: The query tuple containing the values to query for.
        :type values: class: `tuple[Any, ...]`

        :param reference_index: The index which to query.
        :type reference_index: class: `pd.MultiIndex`
        """
        query_dict = self._query_factory.query_tuple_to_dict(self._registry, values, level_name)
        return self._query_level_with_dict(level_name, query_dict, reference_index)

    def _query_with_dict(
        self,
        query_dict: dict[str, dict[str, Any]],
        reference_index: pd.MultiIndex,
    ) -> pd.MultiIndex:
        """Queries a reference index on all levels with the input query dictionary.

        :param query_dict: The dictionary used to construct the query.
        :type query_dict: class: `dict[str, dict[str, Any]]`

        :param reference_index: The index which to query.
        :type reference_index: class: `pd.MultiIndex`
        """
        index = reference_index
        for level_name in self._hierarchy_levels:
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
        reference_index: pd.MultiIndex,
    ) -> MappedLevelIndex:
        """Maps the unique values on a given level to the corresponding indices from a reference index.

        :param level_name: String identifier for the level to query.
        :type level_name: class: `str`

        :param reference_index: The index which to retrieve the unique values of.
        :type reference_index: class: `pd.MultiIndex`
        """
        unique_level_values = self._get_unique_level_values(level_name, reference_index)
        data_dict = {
            values: self._query_level_with_tuple(level_name, values, reference_index) for values in unique_level_values
        }
        return MappedLevelIndex(data_dict)

    def _level_index_to_nested_dict(
        self,
        level_name: str,
        reference_index: pd.MultiIndex,
    ) -> MappedLevelIndex:
        """Recursively creates a nested mapping for each unique values of each level.

        Uses the internal Categorical codes of the MultiIndex sub-levels
        and np.unique for efficient single-pass grouping instead of
        repeated linear scans.

        :param level_name: String identifier for the level to query.
        :type level_name: class: `str`

        :param reference_index: The index which to retrieve the unique values of.
        :type reference_index: class: `pd.MultiIndex`
        """
        self._query_factory.verify_valid_level_name(level_name)
        hierarchy_index = self._hierarchy_levels.index(level_name)
        positions = self._get_sublevel_positions(level_name, reference_index)

        if len(positions) == 1:
            pos = positions[0]
            codes = reference_index.codes[pos]
            categories = reference_index.levels[pos]
            codes_arr = np.asarray(codes)
            unique_codes, first_idx = np.unique(codes_arr, return_index=True)
            order = np.argsort(first_idx)
            sorted_unique = unique_codes[order]
            sorted_first = first_idx[order]
            boundaries = np.append(sorted_first, len(codes_arr))

            groups = {}
            for i, code in enumerate(sorted_unique):
                key = (categories[code],)
                groups[key] = reference_index[boundaries[i]:boundaries[i + 1]]
        else:
            all_codes = np.column_stack([
                np.asarray(reference_index.codes[p]) for p in positions
            ])
            all_cats = [reference_index.levels[p] for p in positions]
            _, first_idx, inverse = np.unique(
                all_codes, axis=0, return_index=True, return_inverse=True,
            )
            order = np.argsort(first_idx)
            sorted_first = first_idx[order]
            boundaries = np.append(sorted_first, len(all_codes))

            groups = {}
            for i, idx in enumerate(order):
                row = all_codes[first_idx[idx]]
                key = tuple(cats[row[j]] for j, cats in enumerate(all_cats))
                groups[key] = reference_index[boundaries[i]:boundaries[i + 1]]

        is_leaf = hierarchy_index == (self.n_hierarchy_levels - 1)
        if is_leaf:
            return MappedLevelIndex(groups)

        next_level_name = self._hierarchy_levels[hierarchy_index + 1]
        return MappedLevelIndex({
            key: self._level_index_to_nested_dict(next_level_name, sub_idx)
            for key, sub_idx in groups.items()
        })

    def query_level_with_dict(
        self,
        level_name: str,
        query_dict: dict[str, Any],
        reference_index: pd.MultiIndex,
    ) -> pd.MultiIndex:
        """Queries a reference index on a given level with the input query dictionary.

        :param level_name: String identifier for the level to query.
        :type level_name: class: `str`

        :param query_dict: The dictionary used to construct the query.
        :type query_dict: class: `dict[str, Any]`

        :param reference_index: The index which to construct the mask on.
        :type reference_index: class: `pd.MultiIndex`
        """
        self._query_factory.verify_level_query_dict(level_name, query_dict)
        query_dict = self._query_factory.prepare_partial_level_query_dict(level_name, query_dict)
        return self._query_level_with_dict(
            level_name,
            query_dict,
            reference_index,
        )

    def query_with_dict(
        self,
        query_dict: dict[str, dict[str, Any]],
        reference_index: pd.MultiIndex,
    ) -> pd.MultiIndex:
        """Queries a reference index on all levels with the input query dictionary.

        :param query_dict: The dictionary used to construct the query.
        :type query_dict: class: `dict[str, dict[str, Any]]`

        :param reference_index: The index which to construct the mask on.
        :type reference_index: class: `pd.MultiIndex`
        """
        self._query_factory.verify_query_dict(query_dict)
        query_dict = self._query_factory.prepare_partial_query_dict(query_dict)
        return self._query_with_dict(query_dict, reference_index)

    def level_index_to_nested_dict(self, level_name: str, reference_index: pd.MultiIndex) -> MappedLevelIndex:
        """Recursively creates a nested mapping for each unique values of each level, starting from the provided one.

        :param level_name: String identifier for the level to query.
        :type level_name: class: `str`

        :param reference_index: The index which to retrieve the unique values of.
        :type reference_index: class: `pd.MultiIndex`
        """
        return self._level_index_to_nested_dict(level_name, reference_index)

    def index_to_nested_dict(
        self,
        reference_index: pd.MultiIndex,
    ) -> MappedLevelIndex:
        """Recursively creates a nested mapping for each unique values of each level, starting from the first one.

        :param reference_index: The index which to retrieve the unique values of.
        :type reference_index: class: `pd.MultiIndex`
        """
        return self._level_index_to_nested_dict(self._hierarchy_levels[0], reference_index)

    @property
    def registry(self) -> dict[str, tuple[str] | None]:
        """Retrieves the registry associated to the indexer."""
        return self._registry

    @property
    def hierarchy_levels(self) -> list[str]:
        """Returns the stack of levels in hierarchical order."""
        return self._hierarchy_levels

    @property
    def n_hierarchy_levels(self) -> int:
        """Returns the number of hierarchical orders."""
        return len(self._hierarchy_levels)

    @property
    def registry_keys(self) -> tuple[str]:
        """Returns the keys of the registry as a tuple."""
        return tuple(self._registry.keys())

    @property
    def query_factory(self) -> QueryFactory:
        """Returns the associated query factory object."""
        return self._query_factory
