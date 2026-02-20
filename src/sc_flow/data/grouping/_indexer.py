from collections.abc import Collection

import numpy as np
import pandas as pd

from sc_flow._constants import BASE_LEVEL_NAME, CONDITION_LEVEL_NAME, GROUP_LEVEL_NAME
from sc_flow._utils import check_sequence_query_against_reference

__all__ = ["HierarchicalIndexer"]


class HierarchicalIndexer:
    """Class for hierarchical indexing of dataframes in subpopulations

    Indexing is done hierarchically along a stack of levels.
    This means that two levels cannot be at the same hierarchy.
    Two hierarchy levels are always assumed, namely the group/context level
    and the condition/perturbation level. It is possible to stack an arbitrary
    number of levels.

    Each level is defined by a set of columns. Each column becomes its own
    sub-level in the resulting MultiIndex, named as (level_name, col).
    Columns are expected to have a Categorical dtype for efficient indexing.
    """

    def __init__(
        self,
        groups_cols: Collection[str] | None = None,
        conditions_cols: Collection[str] | None = None,
    ) -> None:
        """Initializes the hierarchical indexer.

        :param groups_cols: Collection of string identifiers for the
            columns used to define the base groups. When no columns
            are provided (i.e.: :param: `groups_cols` is `None`),
            it creates a dummy placeholder for the base groups.
            When provided, they will be sorted in lexycological order
            to ensure consistency. Defaults to `None`.
        :type groups_cols: class: `Collection[str] | None`

        :param conditions_cols: Collection of string identifiers for the
            columns used to define the base groups. When no columns
            are provided (i.e.: :param: `conditions_cols` is `None`),
            it creates a dummy placeholder for the conditions groups.
            When provided, they will be sorted in lexycological order
            to ensure consistency. Defaults to `None`.
        :type conditions_cols: class: `Collection[str] | None`
        """
        self._groups_cols = [] if groups_cols is None else groups_cols
        self._conditions_cols = [] if conditions_cols is None else conditions_cols

        self._init_registry()

    def _init_registry(self) -> None:
        """Creates registry mapping each level to their respective columns."""
        self._registry: dict[str, tuple[str, ...] | None] = {
            GROUP_LEVEL_NAME: self._groups_cols if self._groups_cols is None else sorted(self._groups_cols),
            CONDITION_LEVEL_NAME: self._conditions_cols
            if self._conditions_cols is None
            else sorted(self._conditions_cols),
        }
        self._hierarchy_levels: list[str] = [GROUP_LEVEL_NAME, CONDITION_LEVEL_NAME]

    def create_index(self, df: pd.DataFrame) -> pd.MultiIndex:
        """Creates a hierarchical index from the input dataframe.

        Each registry column becomes its own sub-level named (level_name, col).
        Columns must have Categorical dtype for efficient code-based indexing.
        Constructs the MultiIndex directly from codes to avoid factorization.

        :param df: The input data frame on which to construct the hierarchical indexing.
        :type df: class: `pd.DataFrame`
        """
        all_levels = [df.index]
        all_codes = [np.arange(len(df))]
        names: list[str | tuple[str, str]] = [BASE_LEVEL_NAME]

        for level_name, level_cols in self._registry.items():
            check_sequence_query_against_reference(
                level_cols, df.columns,
                allow_missing_from_query=True,
                allow_missing_from_reference=False,
            )
            for col in level_cols:
                series = df[col]
                if not hasattr(series, "cat"):
                    raise TypeError(
                        f"Column '{col}' must be Categorical. "
                        f"Cast with: df['{col}'] = df['{col}'].astype('category')"
                    )
                all_levels.append(series.cat.categories)
                all_codes.append(series.cat.codes.values)
                names.append((level_name, col))

        return pd.MultiIndex(levels=all_levels, codes=all_codes, names=names)

    def update_registry(
        self,
        level_name: str,
        level_cols: Collection[str] | None = None,
        level_hierarchy: int = -1,
        allow_override: bool = False,
    ) -> None:
        """Updates registry with new level data.

        :param level_name: The name of the level to be added.
        :type level_name: class: `str`

        :param level_cols: Collection of string identifiers for the
            columns used to define the new level. Defaults to `None`.
        :type level_cols: class: `Collection[str] | None`

        :param allow_override: Whether to allow overriding data for
            already existing levels. If override is not allowed,
            a :class: `ValueError` is raised when attempting to add
            an already existing level. Defaults to `False`.
        :class allow_override: class: `bool`
        """
        # throw value error when the level name is
        # already present and ovveride is not allowed
        if level_name in self._registry and not allow_override:
            msg = f"Level {level_name} already present, cannot override."
            raise ValueError(msg)

        # update registry and hirarchy dict
        self._registry[level_name] = [] if level_cols is None else sorted(level_cols)
        self._hierarchy_levels.insert(level_hierarchy, level_name)

    def reset_registry(self) -> None:
        """Resets the levels registry created at initialization."""
        self._init_registry()

    @property
    def groups_cols(self) -> Collection[str]:
        """Returns the columns used for the group/context level."""
        return self._groups_cols

    @property
    def conditions_cols(self) -> Collection[str]:
        """Returns the columns used for the group/context level."""
        return self._conditions_cols

    @property
    def hierarchy_levels(self) -> list[str]:
        """Returns the stack of levels in hierarchical order."""
        return self._hierarchy_levels

    @property
    def registry(self) -> dict[str, tuple[str] | None]:
        """Retrieves the registry associated to the indexer."""
        return self._registry

    @property
    def registry_keys(self) -> tuple[str]:
        """Returns the keys of the registry as a tuple."""
        return tuple(self.registry.keys())
