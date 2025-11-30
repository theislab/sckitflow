from dataclasses import dataclass

import numpy as np
import pandas as pd

from sc_flow._utils import check_sequence_query_against_reference
from sc_flow.data.grouping._indexer import HierarchicalIndexer

__all__ = ["IndexSelector"]


@dataclass(frozen=True)
class IndexSelector:
    """"""  # noqa

    registry: dict[str, tuple[str, ...] | None]
    hierarchy_levels: list[str]

    @classmethod
    def init_from_indexer(cls, indexer: HierarchicalIndexer) -> "IndexSelector":
        """"""  # noqa
        return cls(
            indexer.registry,
            indexer._hierarchy_levels,
        )

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
        check_sequence_query_against_reference(
            query_dict.keys(), self.registry[level_name], allow_missing_from_query=False
        )

    def _verify_query_dict(
        self,
        query_dict: dict[str, dict[str, str]],
    ) -> None:
        """"""  # noqa
        check_sequence_query_against_reference(query_dict.keys(), self.registry.keys())
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
    def n_hierarchy_levels(self) -> int:
        """"""  # noqa
        return len(self.hierarchy_levels)

    @property
    def registry_keys(self) -> tuple[str]:
        """Returns the keys of the registry as a tuple."""
        return tuple(self.registry.keys())
