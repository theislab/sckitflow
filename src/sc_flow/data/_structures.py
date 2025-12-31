import abc
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field as dc_field
from types import MappingProxyType
from typing import Any, ClassVar, Literal, overload

import numpy as np
import pandas as pd

from sc_flow._types import MappedArray, MappedLevelIndex, TargetCovariatesEncoderCls
from sc_flow.data._mixins import BatchMixin, DataMixin

__all__ = [
    "BaseData",
    "CategoricalData",
    "StateData",
    "MixedTypeData",
    "CouplingData",
    "DistributionData",
    "MatchedData",
    "NestedData",
]


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


@dataclass(frozen=True)
class CategoricalData(BaseData):
    """Container class for categorical data.

    Any categorical data is defined over a set of column, stored in a :class: `pandas.DataFrame`.
    There are two possible ways to represent categorical variables using this container.
    The first one is to pass pre-computed representations, with the :param repr_dict: argument.
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
    repr_dict: Mapping[str, MappedArray] = dc_field(default_factory=lambda: MappingProxyType({}))
    categorical_encoders: Mapping[str, TargetCovariatesEncoderCls] = dc_field(
        default_factory=lambda: MappingProxyType({})
    )

    def __repr__(self) -> str:
        n_obs, n_vars = self.ann_df.shape
        cols = list(self.ann_df.columns)

        repr_keys = list(self.repr_dict.keys())
        encoder_keys = list(self.categorical_encoders.keys())

        return (
            f"{self.__class__.__name__}("
            f"n_obs={n_obs}, "
            f"n_vars={n_vars}, "
            f"columns={cols}, "
            f"repr_dict_keys={repr_keys}, "
            f"categorical_encoders_keys={encoder_keys})"
        )

    def _slice_with_array(
        self,
        idxs: np.ndarray,
    ) -> "CategoricalData":
        ann_df = self.ann_df.iloc[idxs]
        return self.__class__(ann_df, repr_dict=self.repr_dict, categorical_encoders=self.categorical_encoders)

    @property
    def n_obs(self) -> int:
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

    def __post_init__(self) -> None:
        if self.categorical_covariates is not None and self.continuous_covariates is not None:
            n_obs_cat = self.categorical_covariates.ann_df.shape[0]
            n_obs_cont = self.continuous_covariates.n_obs
            if n_obs_cat != n_obs_cont:
                msg = (
                    "Shape mismatch between categorical and continuous covariates. "
                    f"Found {n_obs_cat} observations for categorical covariates and "
                    f"{n_obs_cont} observations for the continuous covariates."
                )
                raise ValueError(msg)

    def __repr__(self) -> str:
        parts = [f"n_obs={self.n_obs}"]

        if self.categorical_covariates is not None:
            cat = self.categorical_covariates
            cols = list(cat.ann_df.columns)
            parts.append(f"categorical(n_vars={len(cols)}, columns={cols})")
        else:
            parts.append("categorical=None")

        if self.continuous_covariates is not None:
            cont = self.continuous_covariates
            keys = list(cont.mapping.keys())
            spatial_dims = {k: v.shape[1] for k, v in cont.mapping.items()}
            parts.append(f"continuous(keys={keys}, spatial_dims={spatial_dims})")
        else:
            parts.append("continuous=None")

        return f"{self.__class__.__name__}(" + ", ".join(parts)

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
        if self.categorical_covariates is not None:
            return self.categorical_covariates.n_obs

        if self.continuous_covariates is not None:
            return self.continuous_covariates.n_obs
        msg = f"{self.__class__.__name__} must contain at least one covariate container."
        raise ValueError(msg)


@dataclass(frozen=True)
class StateData(BaseData):
    """Container class for state data.

    :param X: Array containing the underlying data.
    :type X: class: `np.ndarray`
    """

    X: np.ndarray

    def __repr__(self) -> str:
        shape = self.X.shape
        n_obs = shape[0]
        spatial_dims = shape[1:] if len(shape) > 1 else ()
        return f"{self.__class__.__name__}(n_obs={n_obs}, spatial_dims={spatial_dims})"

    def _slice_with_array(
        self,
        idxs: np.ndarray,
    ) -> "StateData":
        X = self.X[idxs]
        return self.__class__(X)

    @property
    def n_obs(self) -> int:
        return self.X.shape[0]


@dataclass(frozen=True)
class CouplingData(BaseData):
    """Container class for coupling data.

    :param state_lin: Container for the linear term of samples.
    :type state_lin: class: `StateData`

    :param state_quad: Container for the quadratic term of samples.
        Defaults to `None`.
    :type state_quad: class: `StateData | None`
    """

    state_lin: StateData
    state_quad: StateData | None = None

    def __post_init__(self):
        if self.state_quad is not None:
            self.state_lin._assert_same_n_obs(self.state_quad)

    def __repr__(self) -> str:
        parts = [f"n_obs={self.n_obs}"]

        lin_shape = self.state_lin.X.shape
        lin_dims = lin_shape[1:] if len(lin_shape) > 1 else ()
        parts.append(f"linear(spatial_dims={lin_dims})")

        if self.state_quad is not None:
            quad_shape = self.state_quad.X.shape
            quad_dims = quad_shape[1:] if len(quad_shape) > 1 else ()
            parts.append(f"quadratic(spatial_dims={quad_dims})")
        else:
            parts.append("quadratic=None")

        return f"{self.__class__.__name__}({', '.join(parts)})"

    def _slice_with_array(self, idxs: np.ndarray) -> "CouplingData":
        state_lin = self.state_lin.slice_with_array(idxs)
        state_quad = None if self.state_quad is None else self.state_quad.slice_with_array(idxs)
        return self.__class__(
            state_lin,
            state_quad=state_quad,
        )

    def assert_same_spatial_dims(
        self,
        other: "CouplingData",
    ) -> None:
        """Checks that the current coupling data shares the same number of spatial dimensions with another.

        The check is done over the linear term only, as this will be the factor in the product space that
        will be shared by both source and target distributions. The remaining quadratic terms do not need
        to align.
        """
        n_dims_self = self.state_lin.X.shape[1]
        n_dims_other = other.state_lin.X.shape[1]
        if n_dims_self != n_dims_other:
            msg = (
                "Coupling data should share the same number of spatial "
                f"dimensions for the linear term, found {n_dims_self} and {n_dims_other}."
            )
            raise ValueError(msg)

    @classmethod
    def init_from_state_data(
        cls,
        state_data: StateData,
        n_shared_dims: int | None = None,
    ) -> "CouplingData":
        """Initializes the coupling data from base state data and number of shared dimensions.

        :param state_data: The underlying state data.
        :type state_data: class: `StateData`

        :param n_shared_dims: The number of shared dimensions. Defaults to `None` (all dimensions).
        :type n_shared_dims: class: `int | None`
        """
        n_dims = state_data.X.shape[1]
        if n_shared_dims is not None:
            if n_shared_dims >= n_dims:
                msg = (
                    "The number of shared spatial dimensions should "
                    "be strictly smaller the the number of available dimenions. "
                    f"Queried {n_shared_dims} on state data of shape {state_data.X.shape}"
                )
                raise ValueError(msg)
            state_lin = StateData(state_data.X[:, :n_shared_dims])
            state_quad = StateData(state_data.X[:, n_shared_dims:])
        else:
            state_lin = state_data
            state_quad = None
        return cls(state_lin, state_quad)

    @property
    def n_obs(self) -> int:
        return self.state_lin.n_obs


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
    coupling_data: CouplingData | None = None

    def __post_init__(self) -> None:
        if self.target_data is not None:
            self.state_data._assert_same_n_obs(self.target_data)
        if self.condition_data is not None:
            self.state_data._assert_same_n_obs(self.condition_data)
        if self.groups_data is not None:
            self.state_data._assert_same_n_obs(self.groups_data)
        if self.coupling_data is not None:
            self.state_data._assert_same_n_obs(self.coupling_data)

    def __repr__(self) -> str:
        parts = [f"\n * n_obs={self.n_obs}"]
        to_plot = [
            ("state", self.state_data),
            ("target", self.target_data),
            ("condition", self.condition_data),
            ("groups", self.groups_data),
            ("coupling", self.coupling_data),
        ]
        for prefix, comp in to_plot:
            if comp is None:
                parts.append(f"{prefix}={comp}")
            else:
                parts.append(f"{prefix}={comp!r}")
        return f"{self.__class__.__name__}:" + "\n ".join(parts)

    def _slice_with_array(
        self,
        idxs: np.ndarray,
    ) -> "DistributionData":
        state_data = self.state_data.slice_with_array(idxs)
        target_data = None if self.target_data is None else self.target_data.slice_with_array(idxs)
        condition_data = None if self.condition_data is None else self.condition_data.slice_with_array(idxs)
        groups_data = None if self.groups_data is None else self.groups_data.slice_with_array(idxs)
        coupling_data = None if self.coupling_data is None else self.coupling_data.slice_with_array(idxs)
        return self.__class__(
            state_data,
            target_data=target_data,
            condition_data=condition_data,
            groups_data=groups_data,
            coupling_data=coupling_data,
        )

    @property
    def ann_df(self) -> pd.DataFrame:
        """Returns the annotation data frame constructed jointly from condition and groups data."""
        dfs = []
        if self.condition_data and self.condition_data.categorical_covariates:
            dfs.append(self.condition_data.categorical_covariates.ann_df)
        if self.groups_data:
            dfs.append(self.groups_data.ann_df)
        return pd.concat(dfs, axis=1) if dfs else pd.DataFrame()

    @property
    def n_obs(self) -> int:
        return self.state_data.n_obs


@dataclass(frozen=True)
class MatchedData:
    """Container class for matched data."""

    target_distribution: DistributionData
    source_distribution: DistributionData | None = None

    def __post_init__(self) -> None:
        if self.source_distribution is not None:
            if (
                self.target_distribution.coupling_data is not None
                and self.source_distribution.coupling_data is not None
            ):
                self.target_distribution.coupling_data.assert_same_spatial_dims(self.source_distribution.coupling_data)

    def __repr__(self) -> str:
        target_repr = "\n".join("\t" + line for line in repr(self.target_distribution).splitlines())
        parts = [f" * (target) -> {target_repr}"]

        if self.source_distribution is not None:
            source_repr = "\n".join("\t" + line for line in repr(self.source_distribution).splitlines())
            parts.append(f" * (source) -> {source_repr}")

        return f"{self.__class__.__name__}:\n" + "\n".join(parts)

    @property
    def target_distr(self) -> DistributionData:
        """Alias for :attr: `self.target_distribution`."""
        return self.target_distribution

    @property
    def source_distr(self) -> DistributionData | None:
        """Alias for :attr: `self.source_distribution`."""
        return self.source_distribution


@dataclass(frozen=True)
class NestedData(DataMixin):
    """Recursively mapped container for matched data."""

    strict: ClassVar[bool] = False
    required_value_type: ClassVar[type] = DistributionData

    @classmethod
    def init_from_data(
        cls,
        data: DistributionData,
        reference_index: pd.MultiIndex,
        mapped_index: MappedLevelIndex,
        source_key: tuple[Any] | None = None,
    ) -> "NestedData":
        """Initialized the recursive mapping from the input."""
        return cls._init_tree(data, reference_index, mapped_index, source_key)

    @classmethod
    def _init_leaf_node(
        cls,
        data: DistributionData,
        reference_index: pd.MultiIndex,
        mapped_index: MappedLevelIndex,
        source_key: tuple[Any] | None = None,
    ) -> "NestedData":
        if source_key is not None:
            source_idxs = mapped_index.mapping[source_key]
            source_distribution = data.slice_with_index(reference_index, source_idxs)
            rest_idxs = {k: v for k, v in mapped_index.mapping.items() if k != source_key}
        else:
            source_distribution = None
            rest_idxs = mapped_index.mapping
        return cls(
            {
                key: MatchedData(data.slice_with_index(reference_index, value), source_distribution=source_distribution)
                for key, value in rest_idxs.items()
            }
        )

    @classmethod
    def _init_tree(
        cls,
        data: DistributionData,
        reference_index: pd.MultiIndex,
        mapped_index: MappedLevelIndex,
        source_key: tuple[Any] | None = None,
    ) -> "NestedData":
        return cls(
            {
                key: cls._init_leaf_node(data, reference_index, value, source_key)
                if value.is_leaf
                else cls._init_tree(data, reference_index, value, source_key)
                for key, value in mapped_index.mapping.items()
            }
        )
