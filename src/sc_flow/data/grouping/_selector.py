from collections.abc import Collection
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sc_flow.data.grouping._indexer import HierarchicalIndexer

__all__ = ["IndexSelector"]


@dataclass
class IndexSelector:
    """"""  # noqa

    registry: dict[str, tuple[str, ...] | None]
    hierarchy_levels: list[str]

    @classmethod
    def init_from_indexer(cls, indexer: HierarchicalIndexer) -> "IndexSelector":
        """"""  # noqa
        return cls.__init__(
            indexer.registry,
            indexer._hierarchy_levels,
        )

    @staticmethod
    def _check_columns_against_query(
        query: Collection[str],
        reference: Collection[str],
        allow_missing_from_query: bool = True,
        allow_missing_from_reference: bool = False,
    ) -> None:
        """"""  # noqa
        # set logic for overlap
        union = set(query) + set(reference)
        missing_from_reference = union - set(reference)
        missing_from_query = union - set(query)

        # strictly checking that we have all reference columns
        if len(missing_from_query) and not allow_missing_from_query:
            msg = f"The following reference columns are missing from the query: {missing_from_query}"
            raise ValueError(msg)

        # query value not found in reference set
        if len(missing_from_reference) and not allow_missing_from_reference:
            msg = f"The following query columns dont appear in the reference: {missing_from_reference}"
            raise ValueError(msg)

    @staticmethod
    def _parse_query_dict(
        query_dict: dict[str, str],
    ) -> tuple[str, ...]:
        """"""  # noqa
        return tuple(query_dict[cond] for cond in sorted(query_dict.keys()))

    def _get_level_query_mask(
        self,
        level_name: str,
        query_dict: dict[str, str],
        index: pd.MultiIndex,
    ) -> pd.Series | np.ndarray:
        """"""  # noqa
        # sanity checks
        if level_name not in self.registry_keys:
            msg = f"Level {level_name} not found in {self.registry_keys=}."
            raise KeyError(msg)
        if level_name not in index.names:
            msg = f"Level {level_name} not found in {index.names=}."
            raise KeyError(msg)
        self._verify_level_query_dict(level_name, query_dict)

        # retrieving level values and parsing query dict
        level_values = index.get_level_values(level_name)
        query_value = self._parse_query_dict(query_dict)
        return level_values == query_value

    def _prepare_partial_level_query_dict(
        self,
        level_name: str,
        query_dict: dict[str, str],
        index: pd.MultiIndex,
    ) -> dict[str, str]:
        """"""  # noqa
        # retrieving all level columns
        level_cols = self.registry[level_name]  # noqa

        # retrieving queried level columns
        queried_level_cols = sorted(query_dict.keys())  # noqa

        raise NotImplementedError

    def _prepare_partial_query_dict(
        self,
        query_dict: dict[str, dict[str, str]],
        index: pd.MultiIndex,
    ) -> dict[str, str]:
        """"""  # noqa
        raise NotImplementedError

    def _query_level_with_dict(
        self,
        level_name: str,
        query_dict: dict[str, str],
        index: pd.MultiIndex,
    ) -> pd.MultiIndex:
        """"""  # noqa
        mask = self._get_level_query_mask(level_name, query_dict, index)
        return index[mask]

    def _query_with_dict(
        self,
        query_dict: dict[str, dict[str, str]],
        index: pd.MultiIndex,
    ) -> pd.MultiIndex:
        """"""  # noqa
        # verifying dictionary
        self._verify_query_dict(query_dict)

        # iterating over each level in hierarchical order
        for level_name in self.hierarchy_levels:
            # querying current level
            level_query_dict = query_dict[level_name]
            index = self._query_level_with_dict(
                level_name,
                level_query_dict,
                index,
            )
        return index

    def _verify_level_query_dict(
        self,
        level_name: str,
        query_dict: dict[str, str],
    ) -> None:
        """"""  # noqa
        self._check_columns_against_query(query_dict.keys(), self.registry[level_name], allow_missing_from_query=False)

    def _verify_query_dict(
        self,
        query_dict: dict[str, dict[str, str]],
    ) -> None:
        """"""  # noqa
        self._check_columns_against_query(query_dict.keys(), self.registry.keys())
        for level_name, level_query_dict in query_dict.items():
            self._verify_level_query_dict(level_name, level_query_dict)

    def query_level(
        self,
        level_name: str,
        query_dict: dict[str, str],
        index: pd.MultiIndex,
    ) -> pd.MultiIndex:
        """"""  # noqa
        query_dict = self._prepare_partial_level_query_dict(level_name, query_dict)
        return self._query_level_with_dict(
            level_name,
            query_dict,
            index,
        )

    def query(
        self,
        query_dict: dict[str, dict[str, str]],
        index: pd.MultiIndex,
    ) -> pd.MultiIndex:
        """"""  # noqa
        query_dict = self._prepare_partial_query_dict(query_dict)
        return self._query_level_with_dict(
            query_dict,
            index,
        )

    @property
    def hierarchy_levels(self) -> list[str]:
        """"""  # noqa
        return self._hierarchy_levels

    @property
    def n_hierarchy_levels(self) -> int:
        """"""  # noqa
        return len(self.hierarchy_levels)

    @property
    def registry(self) -> dict[str, tuple[str] | None]:
        """Retrieves the registry associated to the indexer."""
        return self._registry

    @property
    def registry_keys(self) -> tuple[str]:
        """Returns the keys of the registry as a tuple."""
        return tuple(self.registry.keys())
