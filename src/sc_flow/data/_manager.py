from collections.abc import Callable, Collection, Mapping
from typing import Any

import numpy as np
import pandas as pd
from anndata import AnnData

from sc_flow._constants import ORIGINAL_INDEX_KEY
from sc_flow._types import TargetCovariatesEncodingId
from sc_flow.data._composite import NestedData
from sc_flow.data._dims_registry import DataDimensionalitiesRegistry
from sc_flow.data._mixins import MappedLevelIndex
from sc_flow.data.containers._categorical import CategoricalData
from sc_flow.data.containers._coupling import CouplingData
from sc_flow.data.containers._distribution import DistributionData
from sc_flow.data.containers._mixed_type import MixedTypeData
from sc_flow.data.containers._state import StateData
from sc_flow.data.grouping._indexer import HierarchicalIndexer
from sc_flow.data.grouping._selector import IndexSelector
from sc_flow.data.schemas import (
    ConditionDataSchema,
    CouplingDataSchema,
    GroupsDataSchema,
    ResponseDataSchema,
    StateDataSchema,
)
from sc_flow.external._context import ExternalModelContext
from sc_flow.preprocessing._preproc import DataPreprocessor
from sc_flow.preprocessing.transforms._base import BaseTransform

__all__ = ["DataManager"]


class DataManager:
    """Class for managing data configurations."""

    def __init__(
        self,
        sample_rep: str | None = None,
        conditions: dict[str, Collection[str]] | None = None,
        conditions_reps: dict[str, str] | None = None,
        conditions_covariates: Collection[str] | None = None,
        control_values_dict: dict[str, str] | None = None,
        target_categorical_covs_dict: Mapping[str, TargetCovariatesEncodingId] | None = None,
        target_continuous_covs: Collection[str] | None = None,
        groups: Collection[str] | None = None,
        groups_reps: dict[str, str] | None = None,
        groups_encoding: dict[str, TargetCovariatesEncodingId | None] | None = None,
        groups_encoding_transform_fn: dict[str, Callable] | None = None,
        groups_encoding_inverse_transform_fn: dict[str, Callable] | None = None,
        n_shared_dims: int | None = None,
        source_rep: str | None = None,
        matched_keys: dict[tuple[Any], tuple[Any]] | None = None,
        allow_paired_settings_on_condition_view: bool = False,
        state_transform: BaseTransform | None = None,
        state_encoder_context: ExternalModelContext | None = None,
        state_decoder_context: ExternalModelContext | None = None,
        state_preproc_repr_name: str | None = None,
        condition_covariates_transform_dict: dict[str, BaseTransform] | None = None,
        condition_covariates_encoder_context_dict: dict[str, ExternalModelContext] | None = None,
        condition_covariates_decoder_context_dict: dict[str, ExternalModelContext] | None = None,
    ) -> None:
        """Initializes the object.

        :param sample_rep: String identifier for the state representation.
            When provided, it should appear as key in `.obsm` attribute of annotated data objects.
            Otherwise, the representation will fall back to `.X`. Defaults to `None`.
        :type sample_rep: class: `str | None`

        :param conditions: Mapping from each condition level to the corresponding columns,
            used to initialize the conditioning data schema. Defaults to `None`.
        :type conditions: class: `dict[str, Collection[str]] | None`

        :param conditions_reps: Mapping from each condition level to the corresponding representation,
            used to initialize the conditioning dat aschema. Defaults to `None`.
        :type conditions_reps: class: `dict[str, str] | None`

        :param conditions_covariates: Collection of continuous condition covariates,
            used to initialize the conditioning data schema. Defaults to `None`.
        :type conditions_covariates: class: `Collection[str] | None`

        :param control_values_dict: Dictionary mapping each condition level to the
            corresponding value used to indicate control observations. Defaults to `None`.
        :type control_values_dict: class: `dict[str, str] | None`

        :param target_categorical_covs_dict: Mapping indicating the encoding used to tranform categorical
            target covariates, used to initialize the target data schema. Defaults to `None`.
        :type target_categorical_covs_dict: class: `Mapping[str, TargetCovariatesEncodingId] | None = None`

        :param target_continuous_covs: Collection of string identifiers for the continuous target covariates,
            used to initialize the target data schema. Defaults to `None`.
        :type target_continuous_covs: class: `Mapping[str, TargetCovariatesEncodingId] | None = None`

        :param groups: Collection of string identifiers for grouping columns, used to initialize the
            grouping data schema. Defaults to `None`.
        :type groups: class: `Collection[str] | None`

        :param groups_reps: Mapping for pre-computed representations of grouping covariates,
            used to initialize the target data schema. Defaults to `None`.
        :type groups_reps: class: `dict[str, str] | None`

        :param groups_encoding: Mapping for tranformations on grouping covariates,
            used to initialize the target data schema. Defaults to `None`.
        :type groups_encoding: class: `dict[str, TargetCovariatesEncodingId | None] | None`

        :param n_shared_dims: The number of shared dimensions to be considered when matching
            distributions over incomparable spaces, used to initialize the
            coupling data schema. Defaults to `None`.
        :type n_shared_dims: class: `int | None`

        :param source_rep: String identifier for the state representation of source states,
            used when matching distributions over incomparable spaces. Used to initialize
            the coupling data schema. Defaults to `None`.
        :type source_rep: class: `str | None`

        :param matched_keys: Optional keys used to identify the source  and
            corresponding target groups in the case of fixed matches.
            When passed, takes precedence over :param: `source_key`.
            Defaults to `None`, in which case falls back to one to many coupling.
        :type matched_keys: class: `dict[tuple[Any], tuple[Any]] | None`

        :param allow_paired_settings_on_condition_view: Whether to allow paired settings
            when viewing the distribution on the conditoin space. Defaults to `False`,
            in which case the condition space view will fall back to the unpaired
            conditional generation setting regardless of the configurations
            set for the state space view
            (namely :param: `control_values_dict` and :param: `matched_keys`).
        :type allow_paired_settings_on_condition_view: class: `bool`

        :param state_transform: The transformation to be applied to the state data.
            Defaults to `None`.
        :type state_transform: class: `BaseTransform | None`

        :param state_encoder_context: The context for optional encoder models of state data.
            Defaults to `None`.
        :type state_encoder_context: class: `ExternalModelContext | None`

        :param state_decoder_context: The context for optional decoder models of state data.
            Defaults to `None`.
        :type state_decoder_context: class: `ExternalModelContext | None`

        :param state_preproc_repr_name: Identifier for the variable names after preprocessing.
            Defaults to `None`.
        :type state_preproc_repr_name: class: `str | None`

        :param condition_covariates_transform_dict: Optional dictionary mapping each continuous
            covariates to their respective transformation object. Defaults to `None`.
        :type condition_covariates_transform_dict: class: `dict[str, BaseTransform] | None`

        :param condition_covariates_encoder_context_dict: Optional dictionary mapping each continuous
            covariates to their respective encoder context. Defaults to `None`.
        :type condition_covariates_encoder_context_dict: class `dict[str, ExternalModelContext] | None`

        :param condition_covariates_decoder_context_dict: Optional dictionary mapping each continuous
            covariates to their respective decoder context. Defaults to `None`.
        :type condition_covariates_decoder_context_dict: class `dict[str, ExternalModelContext] | None`
        """
        self._control_values_dict = control_values_dict
        self._matched_keys = matched_keys
        self._allow_paired_settings_on_condition_view = allow_paired_settings_on_condition_view

        self._state_data_schema = StateDataSchema(sample_rep=sample_rep)
        self._condition_data_schema = ConditionDataSchema(
            conditions=conditions,
            conditions_reps=conditions_reps,
            conditions_covariates=conditions_covariates,
        )
        self._coupling_data_schema = CouplingDataSchema(
            source_rep=source_rep, target_rep=sample_rep, n_shared_dims=n_shared_dims
        )
        self._target_data_schema = ResponseDataSchema(
            categorical_covs_dict=target_categorical_covs_dict,
            continuous_covs=target_continuous_covs,
        )
        self._groups_data_schema = GroupsDataSchema(
            groups=groups,
            groups_reps=groups_reps,
            groups_encoding=groups_encoding,
            groups_encoding_transform_fn=groups_encoding_transform_fn,
            groups_encoding_inverse_transform_fn=groups_encoding_inverse_transform_fn,
        )
        self._indexer = HierarchicalIndexer(
            groups_cols=self._groups_data_schema.groups,
            conditions_cols=self._condition_data_schema.all_condition_cols,
        )
        self._selector = IndexSelector.init_from_indexer(self._indexer)
        self._preproc = DataPreprocessor(
            conditions_covariates=conditions_covariates,
            state_transform=state_transform,
            state_encoder_context=state_encoder_context,
            state_decoder_context=state_decoder_context,
            state_preproc_repr_name=state_preproc_repr_name,
            condition_covariates_transform_dict=condition_covariates_transform_dict,
            condition_covariates_encoder_context_dict=condition_covariates_encoder_context_dict,
            condition_covariates_decoder_context_dict=condition_covariates_decoder_context_dict,
        )

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
        view_on_condition_space: bool = False,
        condition_state_key: str | None = None,
    ) -> pd.Index:
        """Determines feature names based on the state representation used."""
        if view_on_condition_space:
            if condition_state_key is None:
                raise ValueError("condition_state_key required when view_on_condition_space=True")
            sample_rep = condition_state_key
        else:
            sample_rep = self._state_data_schema.sample_rep
            if sample_rep is None and self._preproc.state_preproc is not None:
                sample_rep = self._preproc.state_preproc.repr_name

        # If we have a valid base name, build feature names as "base_0", "base_1", ...
        if sample_rep is not None:
            return pd.Index([f"{sample_rep}_{i}" for i in range(n_features)])

        # Fallback: use original var_names only if the number of features matches.
        # This typically happens when no preprocessing is applied.
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
        view_on_condition_space: bool = False,
        condition_state_key: str | None = None,
        fit_preproc: bool = False,
        apply_transformations: bool = False,
        require_target_state: bool = True,
    ) -> DistributionData:
        if view_on_condition_space and condition_state_key is None:
            raise ValueError("When modeling on the condition space, the state key should be provided")

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

        # preprocessing
        if fit_preproc:
            self._preproc.fit(distribution_data)
        if apply_transformations:
            distribution_data = self._preproc.transform(distribution_data)

        if view_on_condition_space:
            return distribution_data.view_on_condition_space(condition_state_key)
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
        view_on_condition_space: bool = False,
        control_values_dict: dict[str, str] | None = None,
        matched_keys: dict[tuple[Any], tuple[Any]] | None = None,
    ) -> NestedData:
        # optionally allow paired settings on condition space
        if view_on_condition_space and not self._allow_paired_settings_on_condition_view:
            source_key = None
            matched_keys = None
        else:
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
        view_on_condition_space: bool = False,
        control_values_dict: dict[str, str] | None = None,
        matched_keys: dict[tuple[Any], tuple[Any]] | None = None,
    ) -> NestedData:
        """Hierachically splits a distribution data container into matched subpopulations.

        :param data: The distribution data container for the whole population.
        :type data: class: `DistributionData`

        :param view_on_condition_space: Whether to model condiion as states.
            Defaults to `False`.
        :type view_on_condition_space: class: `bool`

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
            view_on_condition_space=view_on_condition_space,
            control_values_dict=control_values_dict,
            matched_keys=matched_keys,
        )

    def get_distribution_data(
        self,
        adata: AnnData,
        view_on_condition_space: bool = False,
        condition_state_key: str | None = None,
        fit_preproc: bool = False,
        apply_transformations: bool = False,
        require_target_state: bool = True,
    ) -> DistributionData:
        """Compiles an annotated data object into a distribution data container.

        :param adata: The annotated data object to compile.
        :type adata: class: `AnnData`

        :param view_on_condition_space: Whether to model condiion as states.
            Defaults to `False`.
        :type view_on_condition_space: class: `bool`

        :param condition_state_key: The key for the continuous condition covariates to be viewed as state
            when :param: `view_on_condition_space` is `True`. This argument is ignored otherwise.
            Defaults to `None`.
        :type condition_state_key: `str | None`

        :param fit_preproc: Whether to fit the preprocessing module on the compiled data. Defaults to `False`.
        :type fit_preproc: class: `bool`

        :param apply_transformations: Whether to apply the preprocessing transformation on the compiled data.
            Defaults to `False`
        :type apply_transformations: class: `bool`

        :param require_target_state: Whether the state representation (`.X` or the configured
            `obsm` sample representation) is required from `adata`. Set to `False` to compile
            only the distribution metadata (condition/group covariates) when no target state
            data is available, e.g. for prediction without target states. Defaults to `True`.
        :type require_target_state: class: `bool`
        """
        return self._get_distribution_data(
            adata,
            view_on_condition_space=view_on_condition_space,
            condition_state_key=condition_state_key,
            fit_preproc=fit_preproc,
            apply_transformations=apply_transformations,
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
        view_on_condition_space: bool = False,
        condition_state_key: str | None = None,
        fit_preproc: bool = False,
        apply_transformations: bool = False,
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

        :param view_on_condition_space: Whether to model condiion as states.
            Defaults to `False`.
        :type view_on_condition_space: class: `bool`

        :param condition_state_key: The key for the continuous condition covariates to be viewed as state
            when :param: `view_on_condition_space` is `True`. This argument is ignored otherwise.
            Defaults to `None`.
        :type condition_state_key: `str | None`

        :param fit_preproc: Whether to fit the preprocessing module on the compiled data. Defaults to `False`.
        :type fit_preproc: class: `bool`

        :param apply_transformations: Whether to apply the preprocessing transformation on the compiled data.
            Defaults to `False`
        :type apply_transformations: class: `bool`
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
            view_on_condition_space=view_on_condition_space,
            condition_state_key=condition_state_key,
            fit_preproc=fit_preproc,
            apply_transformations=apply_transformations,
            require_target_state=require_target_state,
        )
        return self._get_matched_distributions(
            data,
            view_on_condition_space=view_on_condition_space,
            control_values_dict=control_values_dict,
            matched_keys=matched_keys,
        )

    def get_data_dimensionalities(
        self,
        adata: AnnData,
        view_on_condition_space: bool = False,
        condition_state_key: str | None = None,
        fit_preproc: bool = False,
        apply_transformations: bool = False,
    ) -> DataDimensionalitiesRegistry:
        """Registers the data dimensionalities from the input data according to the current schema.

        :param adata: The annotated data object which to extract the dimensionalities from.
        :type adata: class: `AnnData`

        :param view_on_condition_space: Whether to model condiion as states.
            Defaults to `False`.
        :type view_on_condition_space: class: `bool`

        :param condition_state_key: The key for the continuous condition covariates to be viewed as state
            when :param: `view_on_condition_space` is `True`. This argument is ignored otherwise.
            Defaults to `None`.
        :type condition_state_key: `str | None`

        :param fit_preproc: Whether to fit the preprocessing module on the compiled data. Defaults to `False`.
        :type fit_preproc: class: `bool`

        :param apply_transformations: Whether to apply the preprocessing transformation on the compiled data.
            Defaults to `False`
        :type apply_transformations: class: `bool`
        """
        data: DistributionData = self._get_distribution_data(
            adata,
            view_on_condition_space=view_on_condition_space,
            condition_state_key=condition_state_key,
            fit_preproc=fit_preproc,
            apply_transformations=apply_transformations,
        )
        n_features = data.state_data.X.shape[1]
        feature_names = self._get_feature_names(
            adata,
            n_features,
            view_on_condition_space=view_on_condition_space,
            condition_state_key=condition_state_key,
        )
        return self._get_data_dimensionalities(data, feature_names)

    def get_feature_names(
        self, adata: AnnData, view_on_condition_space: bool = False, condition_state_key: str | None = None
    ) -> pd.Index:
        """Registers the feature names from the input data according to the current schema.

        :param adata: The annotated data object which to extract the feature names from.
        :type adata: class: `AnnData`

        :param view_on_condition_space: Whether to model condiion as states.
            Defaults to `False`.
        :type view_on_condition_space: class: `bool`

        :param condition_state_key: The key for the continuous condition covariates to be viewed as state
            when :param: `view_on_condition_space` is `True`. This argument is ignored otherwise.
            Defaults to `None`.
        :type condition_state_key: `str | None`
        """
        n_features = self._get_state_data(adata).X.shape[1]
        return self._get_feature_names(adata, n_features, view_on_condition_space, condition_state_key)

    def unload_preproc(self) -> None:
        """Unload any external models used in preprocessing."""
        self._preproc.unload()

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
    def preproc(self) -> DataPreprocessor:
        """Returns the underlying data preprocessing module."""
        return self._preproc
