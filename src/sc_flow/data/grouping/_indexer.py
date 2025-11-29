from collections.abc import Collection
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sc_flow._constants import CONDITION_LEVEL_NAME, GROUP_LEVEL_NAME

__all__ = ["HierarchicalIndexer"]


@dataclass
class HierarchicalIndexer:
    """Class for hierarchical indexing of dataframes in subpopulations

    As each level is defined by a set of columns, the corresponding index
    will be represented as a tuple containing the respective values for
    each of the columns. When only one column is provided, it still
    creates a single element tuple for consistency.

    :param groups_cols: Collection of string identifiers for the
        columns used to define the base groups. When no columns
        are provided (i.e.: :param: `groups_cols` is `None`),
        it creates a dummy placeholder for the base groups.
        Defaults to `None`.
    :type groups_cols: class: `Collection[str] | None`

    :param conditions_cols: Collection of string identifiers for the
        columns used to define the base groups. When no columns
        are provided (i.e.: :param: `conditions_cols` is `None`),
        it creates a dummy placeholder for the conditions groups.
        Defaults to `None`.
    :type conditions_cols: class: `Collection[str] | None`
    """

    groups_cols: Collection[str] | None = None
    conditions_cols: Collection[str] | None = None

    def __post_init__(self) -> None:
        """"""  # noqa
        self._init_registry()

    def _init_registry(self) -> None:
        """Creates registry mapping each level to their respective columns."""
        self._registry: dict[str, tuple[str, ...] | None] = {
            GROUP_LEVEL_NAME: self.groups_cols if self.groups_cols is None else sorted(self.groups_cols),
            CONDITION_LEVEL_NAME: self.conditions_cols
            if self.conditions_cols is None
            else sorted(self.conditions_cols),
        }
        self._hierarchy_levels: list[str] = [GROUP_LEVEL_NAME, CONDITION_LEVEL_NAME]

    def create_index(self, df: pd.DataFrame) -> pd.MultiIndex:
        """Creates a hierarchical index from the input dataframe."""
        level_frames = {}

        for level_name, level_cols in self._registry.items():
            # constructing empty array and filling it with dummy value
            if level_cols is None:
                dummy_val = f"{level_name}_dummy"
                level_data = np.full((len(df), 1), dummy_val, dtype=object)

            # use level data when provided
            else:
                # sanity check
                self._check_columns_against_query(level_cols, df.columns)
                level_data = df[level_cols].to_numpy()

            # updating levels
            level_frames[level_name] = map(tuple, level_data)

        # constructing multi index
        return pd.MultiIndex.from_frame(pd.DataFrame(level_frames))

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
        self._registry[level_name] = level_cols if level_cols is None else sorted(level_cols)
        self._hierarchy_levels.insert(level_hierarchy, level_name)

    def reset_registry(self) -> None:
        """Resets the levels registry created at initialization."""
        self._init_registry()

    @property
    def hierarchy_levels(self) -> list[str]:
        """"""  # noqa
        return self._hierarchy_levels

    @property
    def registry(self) -> dict[str, tuple[str] | None]:
        """Retrieves the registry associated to the indexer."""
        return self._registry

    @property
    def registry_keys(self) -> tuple[str]:
        """Returns the keys of the registry as a tuple."""
        return tuple(self.registry.keys())
