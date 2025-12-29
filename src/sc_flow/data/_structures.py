import abc
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from dataclasses import field as dc_field
from types import MappingProxyType
from typing import ClassVar, Literal, overload

import numpy as np
import pandas as pd

from sc_flow._types import MappedArray, NestedMappedLevelIndex, TargetCovariatesEncoderCls
from sc_flow.data._mixins import BatchMixin, DataMixin

__all__ = [
    "BaseData",
    "CategoricalData",
    "StateData",
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
        if not reference_index.is_unique:
            raise ValueError("Reference index must be unique.")
        return reference_index.get_indexer(query_index)

    def _assert_same_n_obs(
        self,
        query_container: "BaseData",
    ) -> None:
        n_obs_ref = self.n_obs
        n_obs_query = query_container.n_obs
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
        """"""  # noqa
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
        idxs = np.asarray(idxs)
        return self._slice_with_array(idxs)

    @property
    @abc.abstractmethod
    def n_obs(self) -> int:
        """"""  # noqa
        raise NotImplementedError


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
    repr_dict: dict[str, MappedArray] = dc_field(default_factory=lambda: MappingProxyType({}))
    categorical_encoders: Mapping[str, TargetCovariatesEncoderCls] = dc_field(
        default_factory=lambda: MappingProxyType({})
    )

    def _slice_with_array(
        self,
        idxs: np.ndarray,
    ) -> "CategoricalData":
        ann_df = self.ann_df.iloc[idxs]
        return self.__class__(ann_df, repr_dict=self.repr_dict, categorical_encoders=self.categorical_encoders)

    @property
    def n_obs(self) -> int:
        """Returns the number of observations associated with the data object."""
        return self.ann_df.shape[0]


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

    # def __post_init__(self) -> None:
    #     if self.categorical_covariates is not None and self.continuous_covariates is not None:
    #         n_obs_cat = self.categorical_covariates.ann_df.shape[0]
    #         n_obs_cont = next(iter(self.continuous_covariates.apply(lambda e: e.shape[0]).mapping.values()))
    #         if n_obs_cat != n_obs_cont:
    #             msg = (
    #                 "Shape mismatch between categorical and continuous covariates. "
    #                 f"Found {n_obs_cat} observations for categorical covariates and "
    #                 f"{n_obs_cont} observations for the continuous covariates."
    #             )
    #             raise ValueError(msg)

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

    @property
    def n_obs(self) -> int:
        """Returns the number of observations associated with the data object."""
        if self.categorical_covariates is not None:
            return self.categorical_covariates.n_obs

        if self.continuous_covariates is not None:
            dims = self.continuous_covariates.reference_dims
            if len(dims) > 1:
                msg = f"Continuous covariates should have only one reference dim, found {len(dims)}"
                raise ValueError(msg)
            return dims[0]
        return 0


@dataclass(frozen=True)
class StateData(BaseData):
    """Container class for state data.

    :param X: Array containing the underlying data.
    :type X: class: `np.ndarray`
    """

    X: np.ndarray

    def _slice_with_array(
        self,
        idxs: np.ndarray,
    ) -> "StateData":
        X = self.X[idxs]
        return self.__class__(X)

    @property
    def n_obs(self) -> int:
        """Returns the number of observations associated with the data object."""
        return self.X.shape[0]


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

    def __post_init__(self):
        if self.target_quad is not None:
            self.target_lin._assert_same_n_obs(self.target_quad)

        if self.source_lin is not None:
            n_dims_source = self.source_lin.X.shape[1]
            n_dims_target = self.target_lin.X.shape[1]
            if n_dims_source != n_dims_target:
                msg = (
                    "The linear terms should share the same number of spatial dimensions for "
                    f"source and target, found {n_dims_source} dimensions for source and "
                    f"{n_dims_target} dimensions for target."
                )
                raise ValueError(msg)

            if self.source_quad is not None:
                self.source_lin._assert_same_n_obs(self.source_quad)

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

    @property
    def n_obs_source(self) -> int:
        """Returns the number of observations associated with the data object for source samples."""
        if self.source_lin is not None:
            return self.source_lin.n_obs
        return 0

    @property
    def n_obs_target(self) -> int:
        """Returns the number of observations associated with the data object for target samples."""
        return self.target_lin.n_obs


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

    def __post_init__(self) -> None:
        if self.target_data is not None:
            self.state_data._assert_same_n_obs(self.target_data)
        if self.condition_data is not None:
            self.state_data._assert_same_n_obs(self.condition_data)
        if self.groups_data is not None:
            self.state_data._assert_same_n_obs(self.groups_data)

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
        """"""  # noqa
        dfs = []
        if self.condition_data and self.condition_data.categorical_covariates:
            dfs.append(self.condition_data.categorical_covariates.ann_df)
        if self.groups_data:
            dfs.append(self.groups_data.ann_df)
        return pd.concat(dfs, axis=1) if dfs else pd.DataFrame()

    @property
    def n_obs(self) -> int:
        """Returns the number of observations associated with the data object."""
        return self.state_data.n_obs


@dataclass(frozen=True)
class MatchedData(BaseData):
    """Container class for matched data."""

    target_data: DistributionData
    source_data: DistributionData | None = None
    coupling_data: CouplingData | None = None

    def __post_init__(self):
        if self.coupling_data is not None:
            self.target_data._assert_same_n_obs(self.coupling_data.target_lin)
            if self.coupling_data.target_quad is not None:
                self.target_data._assert_same_n_obs(self.coupling_data.target_quad)

            if self.source_data is not None:
                if self.coupling_data.source_lin is None:
                    msg = "When passing source data, coupling should contain source linear terms."
                    raise ValueError(msg)
                self.source_data._assert_same_n_obs(self.coupling_data.source_lin)
                if self.coupling_data.source_quad is not None:
                    self.source_data._assert_same_n_obs(self.coupling_data.source_quad)

    @property
    def n_obs(self) -> int:
        """Returns the number of observations associated with the data object."""
        return self.target_data.n_obs


@dataclass(frozen=True)
class NestedData(DataMixin):
    """Recursively mapped container for matched data."""

    strict: ClassVar[bool] = False
    required_value_type: ClassVar[type] = DistributionData
    # required_value_type: ClassVar[type] = MatchedData

    @classmethod
    def init_from_data(
        cls,
        data: DistributionData,
        reference_index: pd.MultiIndex,
        mapped_index: NestedMappedLevelIndex,
        n_levels: int,
        conditions: dict[str, Collection[str]] | None,
        control_values_dict: dict[str, str] | None,
    ) -> "NestedData":
        """Initialized the recursive mapping from the input."""
        return cls._get_mapped_data_from_dict(
            data, reference_index, mapped_index, n_levels, conditions, control_values_dict
        )

    @classmethod
    def _get_leaf_mapped_data_from_dict(
        cls,
        data: DistributionData,
        reference_index: pd.MultiIndex,
        mapped_index: NestedMappedLevelIndex,
        conditions: dict[str, Collection[str]] | None,
        control_values_dict: dict[str, str] | None,
        # ) -> MatchedData:
    ) -> DistributionData:
        if not isinstance(data, cls.required_value_type):
            msg = f"Leaf data is expected to be of type {cls.required_value_type}, found {type(data)}."
            raise TypeError(msg)
        data = data.slice_with_index(reference_index, mapped_index)
        return data

    @classmethod
    def _get_mapped_data_from_dict(
        cls,
        # data: MatchedData,
        data: DistributionData,
        reference_index: pd.MultiIndex,
        mapped_index: NestedMappedLevelIndex,
        n_levels: int,
        conditions: dict[str, Collection[str]] | None,
        control_values_dict: dict[str, str] | None,
    ) -> "NestedData":
        out_dict = {}
        for key, value in mapped_index.mapping.items():
            if isinstance(value, NestedMappedLevelIndex):
                out_dict[key] = cls._get_mapped_data_from_dict(
                    data, reference_index, value, n_levels, conditions, control_values_dict
                )
            else:
                out_dict[key] = cls._get_leaf_mapped_data_from_dict(
                    data, reference_index, value, conditions, control_values_dict
                )
        return cls(out_dict)
