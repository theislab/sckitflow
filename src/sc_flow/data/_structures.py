import abc
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field as dc_field

import numpy as np
import pandas as pd

from sc_flow._types import MappedArray, TargetCovariatesEncoderCls
from sc_flow.data._mixins import BatchMixin

__all__ = [
    "BaseData",
    "CategoricalData",
    "StateData",
    "MixedTypeData",
    "MixedTypeData",
    "CouplingData",
    "DistributionData",
]


@dataclass(frozen=True)
class BaseData(abc.ABC):
    """Base class for data containers."""

    @staticmethod
    def _get_query_idxs(
        reference_index: pd.MultiIndex,
        query_index: pd.MultiIndex,
    ) -> np.ndarray:
        """Retrieved the corresponding indices from a given query and a reference index.

        :param reference_index: The reference index.
        :type reference_index: class: `pd.MultiIndex`

        :param query_index: The query index.
        :type query_index: class: `pd.MultiIndex`
        """
        return reference_index.get_indexer(query_index)

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

    def slice_with_array(
        self,
        idxs: np.ndarray,
    ) -> "BaseData":
        """Slices the underlying data with an array.

        :param idxs: The array storing the indices used for slicing.
        :type idxs: class: `np.ndarray`
        """
        return self._slice_with_array(idxs)

    def slice_with_index(
        self, reference_index: pd.MultiIndex, query_index: pd.MultiIndex, return_index_array: bool = False
    ) -> "BaseData | tuple[BaseData, np.ndarray]":
        """Slices the underlying data with a reference and a query index.

        :param reference_index: The reference index.
        :type reference_index: class: `pd.MultiIndex`

        :param query_index: The query index.
        :type query_index: class: `pd.MultiIndex`

        :param return_index_array: Whether to return the array storing the indices. Defaults to `False`.
        :type return_index_array: class: `bool`
        """
        idxs = self._get_query_idxs(reference_index, query_index)
        query_data = self._slice_with_array(idxs)
        if return_index_array:
            return query_data, idxs
        return query_data


@dataclass(frozen=True)
class CategoricalData(BaseData):
    """Container class for categorical data.

    Any categorical data is defined over a set of column, stored in a :class: `pandas.DataFrame`.
    There are two possible ways to represent categorical variables using this container.
    The first one is to pass pre-computed representations, with the :param: `repr_dict` argument.
    Otherwise it is possible to specify some pre-defined encoders to transform the categorical
    values stored in the data frame into suitable representations.

    :param ann_df: The data frame storing the original values.
    :type ann_df: class: `pandas.DataFrame`

    :param repr_dict: Dictionary storing the pre-computed representations.
    :type repr_dict: class: `dict[str, MappedArray]`

    :param categorical_encoders: Mapping storing the covariate encoders class.
    :type categorical_encoders: Mapping[str, TargetCovariatesEncoderCls]
    """

    ann_df: pd.DataFrame
    repr_dict: dict[str, MappedArray] = dc_field(default_factory=lambda: {})
    categorical_encoders: Mapping[str, TargetCovariatesEncoderCls] = dc_field(default_factory=lambda: {})

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        return f"{cls} with {len(self.ann_df)} observations.\n\t Columns: {self.ann_df.columns}"

    def _slice_with_array(
        self,
        idxs: np.ndarray,
    ) -> "CategoricalData":
        ann_df = self.ann_df.iloc[idxs]
        return self.__class__(ann_df, repr_dict=self.repr_dict, categorical_encoders=self.categorical_encoders)


@dataclass(frozen=True)
class MixedTypeData(BaseData):
    """Container class for mixed categorical and continuous covariates.

    :param categorical_covariates: Container storing the categorical covariates.
    :type categorical_covariates: class: `CategoricalData | None`

    :param continuous_covariates: Container storing the continuous covariates.
    :type continuous_covariates: class: `BatchMixin | None`
    """

    categorical_covariates: CategoricalData | None = None
    continuous_covariates: BatchMixin | None = None

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        categorical_covariates = self.categorical_covariates.__repr__()
        continuous_covariates_names = list(self.continuous_covariates.mapping.keys())
        return (
            f"{cls} with components:"
            f"\n\t {categorical_covariates}"
            f"\n\t Continuous covariates {continuous_covariates_names}"
        )

    def _slice_with_array(
        self,
        idxs: np.ndarray,
    ) -> "MixedTypeData":
        categorical_covariates = (
            None if self.categorical_covariates is None else self.categorical_covariates.slice_with_array(idxs)
        )
        continuous_covariates = (
            None if self.continuous_covariates is None else self.continuous_covariates.apply(lambda e: e[idxs])
        )
        return self.__class__(
            categorical_covariates=categorical_covariates, continuous_covariates=continuous_covariates
        )


@dataclass(frozen=True)
class StateData(BaseData):
    """Container class for state data.

    :param X: Array containing the underlying data.
    :type X: class: `np.ndarray`
    """

    X: np.ndarray

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        return f"{cls} of shape {self.X.shape}."

    def _slice_with_array(
        self,
        idxs: np.ndarray,
    ) -> "StateData":
        X = self.X[idxs]
        return self.__class__(X)


@dataclass(frozen=True)
class CouplingData(BaseData):
    """Container class for coupling data.

    :param target_lin: Container for the linear term of samples from the target distribution.
    :type target_lin: class: `StateData`

    :param target_quad: Container for the quadratic term of samples from the target distribution.
        Defaults to `None`.
    :type target_quad: class: `StateData | None`

    :param source_lin: Container for the linear term of samples from the target distribution.
        Defaults to `None`.
    :type source_lin: class: `StateData | None`

    :param source_quad: Container for the quadratic term of samples from the target distribution.
        Defaults to `None`.
    :type source_quad: class: `StateData | None`
    """

    target_lin: StateData
    target_quad: StateData | None = None
    source_lin: StateData | None = None
    source_quad: StateData | None = None

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        n_obs, n_shared_dims = self.target_lin.shape

        return (
            f"{cls} with {n_obs} observations."
            f"\n\t # Shared dimensions: {n_shared_dims}"
            f"\n\t # Target only dims: {self.target_quad.shape[1]}"
            if self.target_quad is not None
            else f"\n\t # Source only dims: {self.source_quad.shape[1]}"
            if self.source_quad is not None
            else ""
        )

    def _slice_with_array(self, idxs) -> "CouplingData":
        target_lin = self.target_lin.slice_with_array(idxs)
        target_quad = None if self.target_quad is None else self.target_quad.slice_with_array(idxs)
        source_lin = None if self.source_lin is None else self.source_lin.slice_with_array(idxs)
        source_quad = None if self.source_quad is None else self.source_quad.slice_with_array(idxs)
        return self.__class__(
            target_lin,
            target_quad=target_quad,
            source_lin=source_lin,
            source_quad=source_quad,
        )


@dataclass(frozen=True)
class DistributionData(BaseData):
    """Container class for distribution data.

    :param state_data: Container storing the state data.
    :type state_data: class: `StateData`

    :param target_data: Container storing the target data.
        Defaults to `None`.
    :type target_data: class: `MixedTypeData | None`

    :param condition_data: Container storing the condition data.
        Defaults to `None`.
    :type condition_data: class: `MixedTypeData | None`

    :param groups_data: Container storing the group data.
        Defaults to `None`.
    :type groups_data: class: `CategoricalData | None`
    """

    state_data: StateData
    target_data: MixedTypeData | None = None
    condition_data: MixedTypeData | None = None
    groups_data: CategoricalData | None = None

    def __repr__(self) -> str:
        cls = self.__class__.__name__

        return (
            f"{cls} with components:"
            f"\n\t {self.state_data.__repr__()}"
            f"\n\t {self.target_data.__repr__()}"
            f"\n\t {self.condition_data.__repr__()}"
            f"\n\t {self.groups_data.__repr__()}"
        )

    def _slice_with_array(
        self,
        idxs: np.ndarray,
    ) -> "DistributionData":
        state_data = self.state_data.slice_with_array(idxs)
        target_data = None if self.target_data is None else self.target_data.slice_with_array(idxs)
        condition_data = None if self.condition_data is None else self.condition_data.slice_with_array(idxs)
        groups_data = None if self.groups_data is None else self.groups_data.slice_with_array(idxs)
        return self.__class__(
            state_data,
            target_data=target_data,
            condition_data=condition_data,
            groups_data=groups_data,
        )

    @property
    def ann_df(self) -> pd.DataFrame:
        """The annotation data frame for the condition and grouping data."""
        conditions_df = self.condition_data.categorical_covariates.ann_df
        groups_df = self.groups_data.ann_df
        return pd.concat((conditions_df, groups_df), axis=1)
