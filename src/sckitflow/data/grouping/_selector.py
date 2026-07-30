from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from sckitflow.data._mixins import MappedLevelIndex
from sckitflow.data.grouping._indexer import HierarchicalIndexer
from sckitflow.data.grouping._query import QueryFactory

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
        self,
        level_name: str,
        reference_index: pd.MultiIndex,
    ) -> list[int]:
        """Returns the positional indices of sub-levels belonging to a hierarchy level."""
        return [i for i, name in enumerate(reference_index.names) if isinstance(name, tuple) and name[0] == level_name]

    def _get_bounds(
        self,
        col: str,
        val: Any,
        level_name: str,
        reference_index: pd.MultiIndex,
        bounds: np.ndarray | slice | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """"""  # noqa
        pos = reference_index.names.index((level_name, col))
        codes = np.asarray(reference_index.codes[pos])
        categories = reference_index.levels[pos]
        target_code = categories.get_loc(val)
        if bounds:
            codes = codes[bounds]
        left = np.searchsorted(codes, target_code, side="left")
        right = np.searchsorted(codes, target_code, side="right")
        return left, right

    def _query_level_with_dict(
        self,
        level_name: str,
        query_dict: dict[str, Any],
        reference_index: pd.MultiIndex,
    ) -> pd.MultiIndex:
        """Queries a reference index on a given level with the input query dictionary.

        Assumes contiguous groups in sorted data; uses searchsorted on integer codes.

        :param level_name: String identifier for the level to query.
        :type level_name: class: `str`

        :param query_dict: The dictionary used to construct the query.
        :type query_dict: class: `dict[str, Any]`

        :param reference_index: The index which to query.
        :type reference_index: class: `pd.MultiIndex`
        """
        self._query_factory.verify_level_query_dict(level_name, query_dict)
        active = {col: val for col, val in query_dict.items() if val != slice(None)}
        if len(active) == 0:
            return reference_index

        if len(active) == 1:
            col, val = next(iter(active.items()))
            left, right = self._get_bounds(col, val, level_name, reference_index)
            return reference_index[left:right]

        # Multi-column: intersect ranges from each column
        left, right = 0, len(reference_index)
        for col, val in active.items():
            sub_left, sub_right = self._get_bounds(col, val, level_name, reference_index, bounds=slice(left, right))
            left = left + sub_left
            right = left + (sub_right - sub_left)
        return reference_index[left:right]

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

    def _level_index_to_nested_dict(
        self,
        level_name: str,
        reference_index: pd.MultiIndex,
    ) -> MappedLevelIndex:
        """Recursively creates a nested mapping for each unique values of each level.

        Operates entirely on the raw integer code arrays from the top-level
        MultiIndex, never materializing sub-MultiIndex objects.  Leaf nodes
        store slice(start, end) that address the original sorted data directly.

        :param level_name: String identifier for the level to query.
        :type level_name: class: `str`

        :param reference_index: The top-level MultiIndex (only read once).
        :type reference_index: class: `pd.MultiIndex`
        """
        positions = self._get_sublevel_positions(level_name, reference_index)
        all_code_arrays = [np.asarray(reference_index.codes[p]) for p in positions]
        all_categories = [reference_index.levels[p] for p in positions]

        hierarchy_index = self._hierarchy_levels.index(level_name)
        return self._nested_dict_from_codes(
            hierarchy_index,
            all_code_arrays,
            all_categories,
            reference_index,
            0,
            len(reference_index),
        )

    def _nested_dict_from_codes(
        self,
        hierarchy_index: int,
        code_arrays: list[np.ndarray],
        categories: list,
        reference_index: pd.MultiIndex,
        lo: int,
        hi: int,
    ) -> MappedLevelIndex:
        """Recursively groups rows [lo, hi) by the current hierarchy level.

        Works on pre-extracted code arrays so no MultiIndex slicing occurs.

        :param hierarchy_index: Position of the current level in the hierarchy.
        :param code_arrays: Integer code arrays for the current level's columns
            (full length, indexed by [lo:hi]).
        :param categories: Corresponding category indices for each code array.
        :param reference_index: The original top-level MultiIndex (used only
            to resolve the next level's positions on first descent).
        :param lo: Start of the row range (inclusive).
        :param hi: End of the row range (exclusive).
        """
        is_leaf = hierarchy_index == (self.n_hierarchy_levels - 1)
        n = hi - lo

        if len(code_arrays) == 0:
            group_spans = [((), lo, hi)]
        elif len(code_arrays) == 1:
            codes_slice = code_arrays[0][lo:hi]
            cats = categories[0]
            if n > 0:
                changes = np.flatnonzero(codes_slice[1:] != codes_slice[:-1]) + 1
                boundaries = np.empty(len(changes) + 2, dtype=np.intp)
                boundaries[0] = 0
                boundaries[1:-1] = changes
                boundaries[-1] = n
            else:
                boundaries = np.array([0, 0], dtype=np.intp)

            group_spans = []
            for i in range(len(boundaries) - 1):
                start = lo + int(boundaries[i])
                end = lo + int(boundaries[i + 1])
                key = (cats[codes_slice[int(boundaries[i])]],)
                group_spans.append((key, start, end))
        else:
            stacked = np.column_stack([c[lo:hi] for c in code_arrays])
            if n > 0:
                diff = np.any(stacked[1:] != stacked[:-1], axis=1)
                changes = np.flatnonzero(diff) + 1
                boundaries = np.empty(len(changes) + 2, dtype=np.intp)
                boundaries[0] = 0
                boundaries[1:-1] = changes
                boundaries[-1] = n
            else:
                boundaries = np.array([0, 0], dtype=np.intp)

            group_spans = []
            for i in range(len(boundaries) - 1):
                start = lo + int(boundaries[i])
                end = lo + int(boundaries[i + 1])
                row = stacked[int(boundaries[i])]
                key = tuple(cats[row[j]] for j, cats in enumerate(categories))
                group_spans.append((key, start, end))

        if is_leaf:
            return MappedLevelIndex({key: slice(start, end) for key, start, end in group_spans})

        next_level_name = self._hierarchy_levels[hierarchy_index + 1]
        next_positions = self._get_sublevel_positions(next_level_name, reference_index)
        next_code_arrays = [np.asarray(reference_index.codes[p]) for p in next_positions]
        next_categories = [reference_index.levels[p] for p in next_positions]

        return MappedLevelIndex(
            {
                key: self._nested_dict_from_codes(
                    hierarchy_index + 1,
                    next_code_arrays,
                    next_categories,
                    reference_index,
                    start,
                    end,
                )
                for key, start, end in group_spans
            }
        )

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
