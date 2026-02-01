import abc
from dataclasses import asdict, dataclass
from typing import Any, Literal, TypeVar, overload

import numpy as np
import pandas as pd

__all__ = ["BaseData"]


TContainer = TypeVar("TContainer", bound="BaseData")


@dataclass(frozen=True)
class BaseData(abc.ABC):
    """Base class for data containers.

    This is an abstract class and should not be instantiated.
    Derived classes must override the `__len__` and `__getitem__`
    methods.
    """

    @abc.abstractmethod
    def __len__(self) -> int:
        """Returns the length of the data container (i.e.: number of observations).

        Must be overridden by derived classes.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def __getitem__(self, idx: np.ndarray | slice) -> "BaseData":
        """Slices the underlying data.

        Must be overridden by derived classes.

        :param idx: The index or slice used to retrieve the elements.
        :type idx: class: `np.ndarray | slice`
        """
        raise NotImplementedError

    @staticmethod
    def _get_query_idxs(
        reference_index: pd.MultiIndex,
        query_index: pd.MultiIndex,
    ) -> np.ndarray:
        """Retrieves the corresponding indices from a given query and a reference index.

        Both query and reference indices must be unique, otherwise a :class: `ValueError` is thrown.

        :param reference_index: The reference index.
        :type reference_index: class: `pd.MultiIndex`

        :param query_index: The query index.
        :type query_index: class: `pd.MultiIndex`
        """
        if not reference_index.is_unique:
            msg = "Reference index must be unique."
            raise ValueError(msg)
        if not query_index.is_unique:
            msg = "Query index must be unique."
            raise ValueError(msg)
        return reference_index.get_indexer(query_index)

    def assert_same_len(
        self,
        other: TContainer,
    ) -> None:
        """Checks that the current object shares the same number of observations as another.

        :param other: The other data container which the length should be shared with.
        :type other: class: `TContainer`
        """
        n_obs_ref = len(self)
        n_obs_query = len(other)
        if n_obs_ref != n_obs_query:
            msg = (
                "Query and reference should share the same number of observations, "
                f"found {n_obs_ref} observations for reference and {n_obs_query} "
                "observations for query."
            )
            raise ValueError(msg)

    @overload
    def slice_with_index(
        self, reference_index: pd.MultiIndex, query_index: pd.MultiIndex, return_index: Literal[False]
    ) -> "BaseData": ...

    @overload
    def slice_with_index(
        self, reference_index: pd.MultiIndex, query_index: pd.MultiIndex, return_index: Literal[True]
    ) -> "tuple[BaseData, np.ndarray]": ...

    def slice_with_index(
        self, reference_index: pd.MultiIndex, query_index: pd.MultiIndex, return_index: bool = False
    ) -> "BaseData | tuple[BaseData, np.ndarray]":
        """Slices the underlying data using reference and query indices.

        Optionally returns the array storing the computed indices.

        :param reference_index: The reference index.
        :type reference_index: class: `pd.MultiIndex`

        :param query_index: The query index.
        :type query_index: class: `pd.MultiIndex`

        :param return_index: Whether to return the `np.ndarray` storing the indices, to avoid recomputing.
            Defaults to `False`.
        :type return_index: class: `bool`
        """
        idxs = self._get_query_idxs(reference_index, query_index)
        if np.any(idxs < 0):
            msg = "Query index contains entries not present in reference index."
            raise KeyError(msg)

        query_data = self[idxs]
        if return_index:
            return query_data, idxs
        return query_data

    def to_dict(self) -> dict[str, Any]:
        """"""  # noqa
        return asdict(self)
