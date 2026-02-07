from collections.abc import Collection
from typing import Any

from sc_flow._utils import check_sequence_query_against_reference

__all__ = ["QueryFactory"]


class QueryFactory:
    """Class for managing queries over hierarchical indices.

    The only functionalities of this class pertain to the creation and validation of queries.
    """

    def __init__(self, registry: dict[str, tuple[str, ...] | None]) -> None:
        """Initializes the query factory from the level registry.

        :param registry: The dictionary containing the registry for the columns
            corresponding to each level.
        :type registry: class: `dict[str, tuple[str, ...] | None]`
        """
        self._registry = registry

    @staticmethod
    def query_dict_to_tuple(
        query_dict: dict[str, Any],
    ) -> tuple[Any]:
        """Converts a query dictionary to a tuple used to slice the index.

        :param query_dict: The dictionary used to construct the query.
        :type query_dict: class: `dict[str, Collection[Any]]`
        """
        return tuple(query_dict[cond] for cond in sorted(query_dict.keys()))

    @staticmethod
    def query_tuple_to_dict(
        reference_dict: dict[str, tuple[str, ...]],
        values: tuple[Any],
        level_name: str,
    ) -> dict[str, Any]:
        """Converts a query tuple for a level to a query dictionary, given a reference dictionary.

        :param reference_dict: The reference dictionary containing the level information.
        :type reference_dict: class: `dict[str, tuple[str, ...]]`

        :param values: The query tuple to convert to a query dictionary.
        :type values: class: `tuple[Any]`

        :param level_name: String identifier for the level to query.
        :type level_name: class: `str`
        """
        if level_name not in reference_dict:
            msg = f"Level {level_name} not found. Available options are {reference_dict.keys()}."
            raise KeyError(msg)
        ref_columns = sorted(reference_dict[level_name])
        if len(values) != len(ref_columns):
            msg = (
                f"The number of provided values is wrong, got {len(values)}"
                f", expected {len(ref_columns)} for level {level_name}"
            )
            raise ValueError(msg)
        return {col: values[idx] for idx, col in enumerate(ref_columns)}

    def verify_valid_level_name(self, level_name: str, reference: Collection[str] | None = None) -> None:
        """Verifies the validity of a level against its registry and optional reference collection of level names.

        :param level_name: String identifier for the level.
        :type level_name: class: `str`

        :param reference: Optional reference set of level identifiers. Defaults to `None`.
        :type reference: `Collection[str] | None`
        """
        if level_name not in self.registry_keys:
            msg = f"Level {level_name} not found in {self.registry_keys=}."
            raise KeyError(msg)
        if reference is not None and level_name not in reference:
            msg = f"Level {level_name} not found in {reference=}."
            raise KeyError(msg)

    def verify_level_query_dict(
        self,
        level_name: str,
        query_dict: dict[str, Any],
    ) -> None:
        """Verifies the validity of a query dictionary against the columns of a given level.

        :param level_name: String identifier for the level to query.
        :type level_name: class: `str`

        :param query_dict: The dictionary used to construct the query.
        :type query_dict: class: `dict[str, Collection[Any]]`
        """
        self.verify_valid_level_name(level_name, reference=self.registry_keys)
        check_sequence_query_against_reference(
            query_dict.keys(),
            self._registry[level_name],
            allow_missing_from_query=True,
            allow_missing_from_reference=False,
        )

    def verify_query_dict(
        self,
        query_dict: dict[str, dict[str, Any]],
    ) -> None:
        """Verifies the validity of a query dictionary against the columns of all levels.

        :param query_dict: The dictionary used to construct the query.
        :type query_dict: class: `dict[str, dict[str, Any]]`
        """
        check_sequence_query_against_reference(query_dict.keys(), self.registry_keys)
        for level_name, level_query_dict in query_dict.items():
            self.verify_level_query_dict(level_name, level_query_dict)

    def prepare_partial_level_query_dict(
        self,
        level_name: str,
        query_dict: dict[str, Any],
    ) -> dict[str, Any]:
        """Prepares a partial query dictionary for a given level.

        Partial queries are intended as query not specifying a conditions
        over all columns associated to a level. To do so, it will add a
        `slice(None)` query to all missing columns.

        :param level_name: String identifier for the level to query.
        :type level_name: class: `str`

        :param query_dict: The dictionary used to construct the query.
        :type query_dict: class: `dict[str, Collection[Any]]`
        """
        level_cols = sorted(self._registry[level_name])
        expanded_query_dict = {}
        for col in level_cols:
            expanded_query_dict[col] = query_dict.get(col, slice(None))
        return expanded_query_dict

    def prepare_partial_query_dict(
        self,
        query_dict: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """Prepares a partial query dictionary for a given level.

        Partial queries are intended as query not specifying a conditions
        over all columns associated to a level. To do so, it will add a
        `slice(None)` query to all missing columns.

        :param level_name: String identifier for the level to query.
        :type level_name: class: `str`

        :param query_dict: The dictionary used to construct the query.
        :type query_dict: class: `dict[str, dict[str, Any]]`
        """
        expanded_query_dict = {}
        for level_name in self.registry_keys:
            expanded_query_dict[level_name] = self.prepare_partial_level_query_dict(
                level_name, query_dict.get(level_name, {})
            )
        return expanded_query_dict

    @property
    def registry(self) -> dict[str, tuple[str, ...] | None]:
        """Returns the registry associated to the query factory."""
        return self._registry

    @property
    def registry_keys(self) -> tuple[str]:
        """Returns the keys of the registry as a tuple."""
        return tuple(self._registry.keys())
