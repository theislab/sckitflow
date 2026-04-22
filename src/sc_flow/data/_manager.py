from collections.abc import Callable, Collection, Mapping
from typing import Any

import pandas as pd
from anndata import AnnData

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
        """
        self._control_values_dict = control_values_dict
        self._state_data_schema: StateDataSchema = self._init_state_data_schema(sample_rep=sample_rep)
        self._condition_data_schema: ConditionDataSchema = self._init_condition_data_schema(
            conditions=conditions,
            conditions_reps=conditions_reps,
            conditions_covariates=conditions_covariates,
        )
        self._coupling_data_schema: CouplingDataSchema = self._init_coupling_data_schema(
            source_rep=source_rep, target_rep=sample_rep, n_shared_dims=n_shared_dims
        )
        self._target_data_schema: ResponseDataSchema = self._init_target_data_schema(
            categorical_covs_dict=target_categorical_covs_dict,
            continuous_covs=target_continuous_covs,
        )
        self._groups_data_schema: GroupsDataSchema = self._init_groups_data_schema(
            groups=groups,
            groups_reps=groups_reps,
            groups_encoding=groups_encoding,
            groups_encoding_transform_fn=groups_encoding_transform_fn,
            groups_encoding_inverse_transform_fn=groups_encoding_inverse_transform_fn,
        )
        self._indexer: HierarchicalIndexer = self._init_indexer(
            groups_cols=self._groups_data_schema.groups,
            conditions_cols=self._condition_data_schema.all_condition_cols,
        )
        self._selector = IndexSelector.init_from_indexer(self._indexer)

    def _init_state_data_schema(
        self,
        sample_rep: str | None = None,
    ) -> StateDataSchema:
        return StateDataSchema(sample_rep=sample_rep)

    def _init_condition_data_schema(
        self,
        conditions: dict[str, Collection[str]] | None = None,
        conditions_reps: dict[str, str] | None = None,
        conditions_covariates: Collection[str] | None = None,
    ) -> ConditionDataSchema:
        return ConditionDataSchema(
            conditions=conditions,
            conditions_reps=conditions_reps,
            conditions_covariates=conditions_covariates,
        )

    def _init_coupling_data_schema(
        self, source_rep: str | None = None, target_rep: str | None = None, n_shared_dims: int | None = None
    ) -> CouplingDataSchema:
        return CouplingDataSchema(source_rep=source_rep, target_rep=target_rep, n_shared_dims=n_shared_dims)

    def _init_target_data_schema(
        self,
        categorical_covs_dict: Mapping[str, TargetCovariatesEncodingId] | None = None,
        continuous_covs: Collection[str] | None = None,
    ) -> ResponseDataSchema:
        return ResponseDataSchema(
            categorical_covs_dict=categorical_covs_dict,
            continuous_covs=continuous_covs,
        )

    def _init_groups_data_schema(
        self,
        groups: Collection[str] | None = None,
        groups_reps: dict[str, str] | None = None,
        groups_encoding: dict[str, TargetCovariatesEncodingId] | None = None,
        groups_encoding_transform_fn: dict[str, Callable] | None = None,
        groups_encoding_inverse_transform_fn: dict[str, Callable] | None = None,
    ) -> GroupsDataSchema:
        return GroupsDataSchema(
            groups=groups,
            groups_reps=groups_reps,
            groups_encoding=groups_encoding,
            groups_encoding_transform_fn=groups_encoding_transform_fn,
            groups_encoding_inverse_transform_fn=groups_encoding_inverse_transform_fn,
        )

    def _init_indexer(
        self,
        groups_cols: Collection[str] | None = None,
        conditions_cols: Collection[str] | None = None,
    ) -> HierarchicalIndexer:
        return HierarchicalIndexer(
            groups_cols=groups_cols,
            conditions_cols=conditions_cols,
        )

    def _get_feature_names(self, adata: AnnData) -> pd.Index:
        """Determines feature names based on the state representation used."""
        sample_rep = self._state_data_schema.sample_rep
        if sample_rep is None:
            return adata.var_names

        # Case: Latent representation in .obsm (e.g., 'X_pca')
        rep_data = adata.obsm[sample_rep]
        n_features = rep_data.shape[1]

        return pd.Index([f"{sample_rep}_{i}" for i in range(n_features)])

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
    ) -> DistributionData:
        state_data: StateData = self._get_state_data(adata)
        condition_data: MixedTypeData = self._get_condition_data(adata)
        response_data: MixedTypeData = self._get_target_data(adata)
        groups_data: CategoricalData = self._get_groups_data(adata)
        source_coupling_data, target_coupling_data = self._get_coupling_data(adata)
        return DistributionData(
            state_data,
            response_data=response_data,
            condition_data=condition_data,
            groups_data=groups_data,
            source_coupling_data=source_coupling_data,
            target_coupling_data=target_coupling_data,
        )

    def _get_matched_distributions(
        self,
        data: DistributionData,
    ) -> NestedData:
        index: pd.DataFrame = self._indexer.create_index(data.ann_df)
        mapped_index: MappedLevelIndex = self._selector.index_to_nested_dict(index)
        return NestedData.init_from_data(data, index, mapped_index, source_key=self.source_key)

    def _get_data_dimensionalities(
        self,
        data: DistributionData,
        feature_names: pd.Index,
    ) -> DataDimensionalitiesRegistry:
        return DataDimensionalitiesRegistry.init_from_distribution_data(data, feature_names)

    def get_matched_distributions(
        self,
        data: DistributionData,
    ) -> NestedData:
        """Hierachically splits a distribution data container into matched subpopulations.

        :param data: The distribution data container for the whole population.
        :type data: class: `DistributionData`
        """
        return self._get_matched_distributions(data)

    def get_distribution_data(
        self,
        adata: AnnData,
    ) -> DistributionData:
        """Compiles an annotated data object into a distribution data container.

        :param adata: The annotated data object to compile.
        :type adata: class: `AnnData`
        """
        return self._get_distribution_data(adata)

    def compile_adata(
        self,
        adata: AnnData,
    ) -> NestedData:
        """Compiles the annotated data objects and returns the split and matched subpopulations.

        :param adata: The annotated data object to compile and split.
        :type adata: class: `AnnData`
        """
        data: DistributionData = self._get_distribution_data(adata)
        return self._get_matched_distributions(data)

    def get_data_dimensionalities(
        self,
        adata: AnnData,
    ) -> DataDimensionalitiesRegistry:
        """Registers the data dimensionalities from the input data according to the current schema.

        :param adata: The annotated data object which to extract the dimensionalities from.
        :type adata: class: `AnnData`
        """
        data: DistributionData = self._get_distribution_data(adata)
        feature_names = self._get_feature_names(adata)
        return self._get_data_dimensionalities(data, feature_names)

    def get_feature_names(
        self,
        adata: AnnData,
    ) -> pd.Index:
        """Registers the feature names from the input data according to the current schema.

        :param adata: The annotated data object which to extract the feature names from.
        :type adata: class: `AnnData`
        """
        return self._get_feature_names(adata)

    @property
    def control_values_dict(self) -> dict[str, str] | None:
        """Exposes the homonymous attribute set at initialization."""
        return self._control_values_dict

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
        if self._control_values_dict is None:
            return None
        control_query_dict = {
            cond: self._control_values_dict[level]
            for level, conditions in self._condition_data_schema.conditions.items()
            for cond in conditions
        }
        return self._selector.query_factory.query_dict_to_tuple(control_query_dict)
