import abc
from dataclasses import dataclass
from typing import Literal, overload

import numpy as np
import pandas as pd

__all__ = ["BaseData"]


@dataclass(frozen=True)
class BaseData(abc.ABC):
    """Base class for data containers."""

    def __len__(self) -> int:
        return self.n_obs

    @staticmethod
    def _get_query_idxs(
        reference_index: pd.MultiIndex,
        query_index: pd.MultiIndex,
    ) -> np.ndarray:
        """Retrieves the corresponding indices from a given query and a reference index.

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

    def _assert_same_n_obs(
        self,
        other: "BaseData",
    ) -> None:
        """Checks that the current object shares the same number of observations as another."""
        n_obs_ref = self.n_obs
        n_obs_query = other.n_obs
        if n_obs_ref != n_obs_query:
            msg = (
                "Query and reference should share the same number of observations, "
                f"found {n_obs_ref} observations for reference and {n_obs_query} "
                "observations for query."
            )
            raise ValueError(msg)

    @abc.abstractmethod
    def _slice_with_array(
        self,
        idxs: np.ndarray,
    ) -> "BaseData":
        """Slices the underlying data with an array.

        Needs to be overridden by children classes.

        :param idxs: The array storing the indices used for slicing.
        :type idxs: class: `np.ndarray`
        """
        raise NotImplementedError

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

        query_data = self._slice_with_array(idxs)
        if return_index:
            return query_data, idxs
        return query_data

    def slice_with_array(
        self,
        idxs: np.ndarray,
    ) -> "BaseData":
        """Slices the underlying data with an array.

        :param idxs: The array storing the indices used for slicing.
        :type idxs: class: `np.ndarray`
        """
        if not isinstance(idxs, np.ndarray):
            idxs = np.asarray(idxs, dtype=int)
        return self._slice_with_array(idxs)

    @property
    @abc.abstractmethod
    def n_obs(self) -> int:
        raise NotImplementedError
