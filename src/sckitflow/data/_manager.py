from collections.abc import Collection, Mapping
from typing import Any, TypedDict, Unpack

import numpy as np
import pandas as pd
from anndata import AnnData

from sckitflow._constants import ORIGINAL_INDEX_KEY
from sckitflow._types import TargetCovariatesEncodingId
from sckitflow.data._composite import NestedData
from sckitflow.data._dims_registry import DataDimensionalitiesRegistry
from sckitflow.data._group_encoders import GroupEncoder, GroupEncoderId
from sckitflow.data._mixins import MappedLevelIndex
from sckitflow.data.containers._categorical import CategoricalData
from sckitflow.data.containers._coupling import CouplingData
from sckitflow.data.containers._distribution import DistributionData
from sckitflow.data.containers._mixed_type import MixedTypeData
from sckitflow.data.containers._state import StateData
from sckitflow.data.grouping._indexer import HierarchicalIndexer
from sckitflow.data.grouping._selector import IndexSelector
from sckitflow.data.schemas import (
    ConditionDataSchema,
    CouplingDataSchema,
    GroupsDataSchema,
    ResponseDataSchema,
    StateDataSchema,
)

__all__ = ["DataManagerKwargs", "DataManager"]


class DataManagerKwargs(TypedDict, total=False):
    """Keyword arguments accepted by :class:`DataManager`."""

    sample_rep: str | None
    """String identifier for the state representation. When provided, it should appear as key in
    `.obsm` attribute of annotated data objects. Otherwise, the representation will fall back to
    `.X`. Defaults to `None`."""

    conditions: dict[str, Collection[str]] | None
    """Mapping from each condition level to the corresponding columns, used to initialize the
    conditioning data schema. Defaults to `None`."""

    conditions_reps: dict[str, str] | None
    """Mapping from each condition level to the corresponding representation, used to initialize the
    conditioning data schema. Defaults to `None`."""

    conditions_covariates: Collection[str] | None
    """Collection of continuous condition covariates, used to initialize the conditioning data
    schema. Defaults to `None`."""

    control_values_dict: dict[str, str] | None
    """Dictionary mapping each condition level to the corresponding value used to indicate control
    observations. Defaults to `None`."""

    condition_state_key: str | None
    """The key for the continuous condition covariates to be viewed as state when
    `view_on_condition_space` is `True`. This argument is ignored otherwise. Defaults to `None`."""

    target_categorical_covs_dict: Mapping[str, TargetCovariatesEncodingId] | None
    """Mapping indicating the encoding used to transform categorical target covariates, used to
    initialize the target data schema. Defaults to `None`."""

    target_continuous_covs: Collection[str] | None
    """Collection of string identifiers for the continuous target covariates, used to initialize the
    target data schema. Defaults to `None`."""

    groups: Collection[str] | None
    """Collection of string identifiers for grouping columns, used to initialize the grouping data
    schema. Defaults to `None`."""

    groups_reps: dict[str, str] | None
    """Mapping for pre-computed representations of grouping covariates, used to initialize the target
    data schema. Defaults to `None`."""

    groups_encoding: dict[str, GroupEncoder | GroupEncoderId] | None
    """Mapping from each group column to a :class:`~sckitflow.data._group_encoders.GroupEncoder`
    (e.g. ``OneHot()``, ``Label()``, ``Affine(scale=2.0)``), used to initialize the grouping data
    schema. Encoders are serializable dataclasses that build their fitted transformer on demand. The
    string ids ``"label"`` / ``"one-hot"`` are accepted as shorthand for the parameter-free encoders.
    Defaults to `None`."""

    n_shared_dims: int | None
    """The number of shared dimensions to be considered when matching distributions over
    incomparable spaces, used to initialize the coupling data schema. Defaults to `None`."""

    source_rep: str | None
    """String identifier for the state representation of source states, used when matching
    distributions over incomparable spaces. Used to initialize the coupling data schema.
    Defaults to `None`."""

    matched_keys: dict[tuple[Any], tuple[Any]] | None
    """Optional keys used to identify the source and corresponding target groups in the case of
    fixed matches. When passed, takes precedence over `source_key`. Defaults to `None`, in which
    case falls back to one-to-many coupling."""


class DataManager:
    """Class for managing data configurations."""

    def __init__(self, **kwargs: Unpack[DataManagerKwargs]) -> None:
        """Initializes the object. See :class:`DataManagerKwargs` for parameter descriptions."""
        self._control_values_dict = kwargs.get("control_values_dict")
        self._condition_state_key = kwargs.get("condition_state_key")
        self._matched_keys = kwargs.get("matched_keys")

        self._state_data_schema = StateDataSchema(sample_rep=kwargs.get("sample_rep"))
        self._condition_data_schema = ConditionDataSchema(
            conditions=kwargs.get("conditions"),
            conditions_reps=kwargs.get("conditions_reps"),
            conditions_covariates=kwargs.get("conditions_covariates"),
        )
        self._coupling_data_schema = CouplingDataSchema(
            source_rep=kwargs.get("source_rep"),
            target_rep=kwargs.get("sample_rep"),
            n_shared_dims=kwargs.get("n_shared_dims"),
        )
        self._target_data_schema = ResponseDataSchema(
            categorical_covs_dict=kwargs.get("target_categorical_covs_dict"),
            continuous_covs=kwargs.get("target_continuous_covs"),
        )
        self._groups_data_schema = GroupsDataSchema(
            groups=kwargs.get("groups"),
            groups_reps=kwargs.get("groups_reps"),
            groups_encoding=kwargs.get("groups_encoding"),
        )
        self._indexer = HierarchicalIndexer(
            groups_cols=self._groups_data_schema.groups,
            conditions_cols=self._condition_data_schema.all_condition_cols,
        )
        self._selector = IndexSelector.init_from_indexer(self._indexer)

    @property
    def _view_on_condition_space(self) -> bool:
        return self._condition_state_key is not None

    def _get_source_key(
        self,
        control_values_dict: dict[str, str] | None = None,
    ) -> tuple[Any] | None:
        if control_values_dict is None:
            return None
        control_query_dict = {
            cond: control_values_dict[level]
            for level, conditions in self._condition_data_schema.conditions.items()
            for cond in conditions
        }
        return self._selector.query_factory.query_dict_to_tuple(control_query_dict)

    def _get_feature_names(
        self,
        adata: AnnData,
        n_features: int,
    ) -> pd.Index:
        """Determines feature names based on the state representation used."""
        if self._view_on_condition_space:
            sample_rep = self._condition_state_key
        else:
            sample_rep = self._state_data_schema.sample_rep

        # If we have a valid base name, build feature names as "base_0", "base_1", ...
        if sample_rep is not None:
            return pd.Index([f"{sample_rep}_{i}" for i in range(n_features)])

        # Fallback: use original var_names only if the number of features matches.
        if n_features == adata.shape[-1]:
            return adata.var_names

        # Last resort: generic feature names.
        return pd.Index([f"feature_{i}" for i in range(n_features)])

    def _get_state_data(
        self,
        adata: AnnData,
    ) -> StateData:
        return self._state_data_schema.get_data(adata)

    def _get_condition_data(
        self,
        adata: AnnData,
    ) -> MixedTypeData | None:
        return self._condition_data_schema.get_data(adata)

    def _get_coupling_data(self, adata: AnnData) -> tuple[CouplingData, CouplingData]:
        return self._coupling_data_schema.get_data(adata)

    def _get_groups_data(
        self,
        adata: AnnData,
    ) -> CategoricalData:
        return self._groups_data_schema.get_data(adata)

    def _get_target_data(
        self,
        adata: AnnData,
    ) -> MixedTypeData | None:
        return self._target_data_schema.get_data(adata)

    def _get_distribution_data(
        self,
        adata: AnnData,
        require_target_state: bool = True,
    ) -> DistributionData:
        state_data: StateData | None = self._get_state_data(adata) if require_target_state else None
        condition_data: MixedTypeData = self._get_condition_data(adata)
        response_data: MixedTypeData = self._get_target_data(adata)
        groups_data: CategoricalData = self._get_groups_data(adata)
        if require_target_state:
            source_coupling_data, target_coupling_data = self._get_coupling_data(adata)
        else:
            # coupling reps default to the state representation (`.X`) unless explicitly
            # configured otherwise, so they are unavailable in the same cases as state_data.
            source_coupling_data, target_coupling_data = None, None

        distribution_data = DistributionData(
            state_data,
            response_data=response_data,
            condition_data=condition_data,
            groups_data=groups_data,
            source_coupling_data=source_coupling_data,
            target_coupling_data=target_coupling_data,
        )

        if self._view_on_condition_space:
            return distribution_data.view_on_condition_space(self._condition_state_key)
        return distribution_data

    def _get_mapped_index(self, ann_df: pd.DataFrame) -> MappedLevelIndex:
        index: pd.MultiIndex = self._indexer.create_index(ann_df)
        mapped_index: MappedLevelIndex = self._selector.index_to_nested_dict(index)
        return mapped_index

    @staticmethod
    def _assert_sorted(data: DistributionData) -> None:
        if not data.is_sorted:
            raise ValueError(
                "DistributionData is not sorted by its annotation columns. "
                "Either pass sort=True to compile_adata() or call "
                "DataManager.sort_adata(adata) before compilation."
            )

    def _get_matched_distributions(
        self,
        data: DistributionData,
        control_values_dict: dict[str, str] | None = None,
        matched_keys: dict[tuple[Any], tuple[Any]] | None = None,
    ) -> NestedData:
        # ---- Get control values and matched keys ----
        if control_values_dict:
            source_key = self._get_source_key(control_values_dict)
        else:
            source_key = self.source_key

        if matched_keys is None:
            matched_keys = self.matched_keys

        self._assert_sorted(data)
        mapped_index = self._get_mapped_index(data.ann_df)
        return NestedData.init_from_data(
            data,
            mapped_index,
            source_key=source_key,
            matched_keys=matched_keys,
        )

    def _get_data_dimensionalities(
        self,
        data: DistributionData,
        feature_names: pd.Index,
    ) -> DataDimensionalitiesRegistry:
        return DataDimensionalitiesRegistry.init_from_distribution_data(data, feature_names)

    def get_matched_distributions(
        self,
        data: DistributionData,
        control_values_dict: dict[str, str] | None = None,
        matched_keys: dict[tuple[Any], tuple[Any]] | None = None,
    ) -> NestedData:
        """Hierachically splits a distribution data container into matched subpopulations.

        :param data: The distribution data container for the whole population.
        :type data: class: `DistributionData`

        :param control_values_dict: Optional dictionary mapping each condition
            level to the corresponding value used to indicate control observations.
            This overrides the homonimous attribute and is needed to allow
            inference over arbitrary control keys at inference time.
            Without this, inference would be bound to the source
            group defined for training. Defaults to `None`,
            in which case the instance attribute will be used.
        :type control_values_dict: class: `dict[str, str] | None`

        :param matched_keys: Optional keys used to identify the source  and
            corresponding target groups in the case of fixed matches.
            This overrides the homonimous attribute and is needed to allow
            inference over arbitrary matched groups at inference time.
            Without this, inference would be bound to the pairs of source
            and target groups defined for training. Defaults to `None`,
            in which case the instance attribute will be used.
        :type matched_keys: class: `dict[tuple[Any], tuple[Any]] | None`
        """
        return self._get_matched_distributions(
            data,
            control_values_dict=control_values_dict,
            matched_keys=matched_keys,
        )

    def get_distribution_data(
        self,
        adata: AnnData,
        require_target_state: bool = True,
    ) -> DistributionData:
        """Compiles an annotated data object into a distribution data container.

        :param adata: The annotated data object to compile.
        :type adata: class: `AnnData`

        :param require_target_state: Whether the state representation (`.X` or the configured
            `obsm` sample representation) is required from `adata`. Set to `False` to compile
            only the distribution metadata (condition/group covariates) when no target state
            data is available, e.g. for prediction without target states. Defaults to `True`.
        :type require_target_state: class: `bool`
        """
        return self._get_distribution_data(
            adata,
            require_target_state=require_target_state,
        )

    def sort_adata(self, adata: AnnData) -> AnnData:
        """Sort an AnnData by the hierarchy columns (groups then conditions).

        Returns a **new** AnnData with rows reordered.  AnnData does not
        support true in-place reordering of ``.X``, ``.obsm``, etc., so a
        sorted copy is returned instead.  The original object is never
        mutated.

        The original ``obs_names`` are preserved in
        ``adata.obs[ORIGINAL_INDEX_KEY]`` on the returned object so
        they can be recovered later.

        :param adata: The annotated data object to sort.
        :type adata: class: `AnnData`

        :returns: A new AnnData with rows sorted by the hierarchy columns.
        :rtype: class: `AnnData`
        """
        sort_cols = list(self._indexer.sort_columns)
        if not sort_cols:
            return adata

        sort_keys = [adata.obs[c] for c in reversed(sort_cols)]
        order = np.lexsort(sort_keys)

        sorted_adata = adata[order].copy()
        sorted_adata.obs[ORIGINAL_INDEX_KEY] = adata.obs_names[order].values
        sorted_adata.obs_names = pd.RangeIndex(len(sorted_adata)).astype(str)
        return sorted_adata

    def compile_adata(
        self,
        adata: AnnData,
        sort: bool = False,
        control_values_dict: dict[str, str] | None = None,
        matched_keys: dict[tuple[Any], tuple[Any]] | None = None,
        require_target_state: bool = True,
    ) -> NestedData:
        """Compile an annotated data object into split and matched subpopulations.

        :param adata: The annotated data object to compile and split.
        :type adata: class: `AnnData`

        :param sort: If ``True``, create a sorted copy of *adata* via
            :meth:`sort_adata` and compile from that copy.  When setting this one
            should keep in mind this will copy the full adata object. When ``False`` (the
            default) the data must already be sorted; a ``ValueError``
            is raised otherwise.
        :type sort: class: `bool`

        :param control_values_dict: Optional dictionary mapping each condition
            level to the corresponding value used to indicate control observations.
            This overrides the homonimous attribute and is needed to allow
            inference over arbitrary control keys at inference time.
            Without this, inference would be bound to the source
            group defined for training. Defaults to `None`,
            in which case the instance attribute will be used.
        :type control_values_dict: class: `dict[str, str] | None`

        :param matched_keys: Optional keys used to identify the source  and
            corresponding target groups in the case of fixed matches.
            This overrides the homonimous attribute and is needed to allow
            inference over arbitrary matched groups at inference time.
            Without this, inference would be bound to the pairs of source
            and target groups defined for training. Defaults to `None`,
            in which case the instance attribute will be used.
        :type matched_keys: class: `dict[tuple[Any], tuple[Any]] | None`

        :param require_target_state: Whether the state representation (`.X` or the configured
            `obsm` sample representation) is required from `adata`. Set to `False` to compile
            only the distribution metadata (condition/group covariates) when no target state
            data is available, e.g. for prediction without target states. Defaults to `True`.
        :type require_target_state: class: `bool`
        """
        if sort:
            adata = self.sort_adata(adata)
        data: DistributionData = self._get_distribution_data(
            adata,
            require_target_state=require_target_state,
        )
        return self._get_matched_distributions(
            data,
            control_values_dict=control_values_dict,
            matched_keys=matched_keys,
        )

    def get_data_dimensionalities(
        self,
        adata: AnnData,
    ) -> DataDimensionalitiesRegistry:
        """Registers the data dimensionalities from the input data according to the current schema.

        :param adata: The annotated data object which to extract the dimensionalities from.
        :type adata: class: `AnnData`
        """
        data: DistributionData = self._get_distribution_data(adata)
        n_features = data.state_data.X.shape[1]
        feature_names = self._get_feature_names(adata, n_features)
        return self._get_data_dimensionalities(data, feature_names)

    def get_feature_names(self, adata: AnnData) -> pd.Index:
        """Registers the feature names from the input data according to the current schema.

        :param adata: The annotated data object which to extract the feature names from.
        :type adata: class: `AnnData`
        """
        n_features = self._get_state_data(adata).X.shape[1]
        return self._get_feature_names(adata, n_features)

    @property
    def control_values_dict(self) -> dict[str, str] | None:
        """Exposes the homonymous attribute set at initialization."""
        return self._control_values_dict

    @property
    def matched_keys(self) -> dict[tuple[Any], tuple[Any]] | None:
        """Exposes the homonymous attribute set at initialization."""
        return self._matched_keys

    @property
    def indexer(self) -> HierarchicalIndexer:
        """Returns the indexer used to compute the hierarchical splits."""
        return self._indexer

    @property
    def selector(self) -> IndexSelector:
        """Returns the selector used to slice with the hierarchical splits."""
        return self._selector

    @property
    def state_data_schema(self) -> StateDataSchema:
        """Exposes the state data schema."""
        return self._state_data_schema

    @property
    def condition_data_schema(self) -> ConditionDataSchema:
        """Exposes the condition data schema."""
        return self._condition_data_schema

    @property
    def coupling_data_schema(self) -> CouplingDataSchema:
        """Exposes the coupling data schema."""
        return self._coupling_data_schema

    @property
    def groups_data_schema(self) -> GroupsDataSchema:
        """Exposes the coupling data schema."""
        return self._groups_data_schema

    @property
    def target_data_schema(self) -> ResponseDataSchema:
        """Exposes the target data schema."""
        return self._target_data_schema

    @property
    def source_key(self) -> tuple[Any] | None:
        """Returns the key used to define the source subpopulations."""
        return self._get_source_key(self._control_values_dict)

    @property
    def condition_state_key(self) -> str | None:
        """Returns the key used to define the condition covariates to be viewed as state."""
        return self._condition_state_key
