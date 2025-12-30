from collections.abc import Collection, Mapping
from typing import Any

import pandas as pd
from anndata import AnnData

from sc_flow._types import NestedMappedLevelIndex, TargetCovariatesEncodingId
from sc_flow.data._structures import (
    CategoricalData,
    DistributionData,
    MixedTypeData,
    NestedData,
    StateData,
)
from sc_flow.data.grouping._indexer import HierarchicalIndexer
from sc_flow.data.grouping._selector import IndexSelector
from sc_flow.data.schemas import ConditionDataSchema, GroupsDataSchema, StateDataSchema, TargetDataSchema

__all__ = ["DataManager"]


class DataManager:
    """"""  # noqa

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
    ) -> None:
        """"""  # noqa

        self._control_values_dict = control_values_dict
        self._state_data_schema: StateDataSchema = self._init_state_data_schema(sample_rep=sample_rep)
        self._condition_data_schema: ConditionDataSchema = self._init_condition_data_schema(
            conditions=conditions,
            conditions_reps=conditions_reps,
            conditions_covariates=conditions_covariates,
        )
        self._target_data_schema: TargetDataSchema = self._init_target_data_schema(
            categorical_covs_dict=target_categorical_covs_dict,
            continuous_covs=target_continuous_covs,
        )
        self._groups_data_schema: GroupsDataSchema = self._init_groups_data_schema(
            groups=groups,
            groups_reps=groups_reps,
            groups_encoding=groups_encoding,
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
        """"""  # noqa
        return StateDataSchema(sample_rep=sample_rep)

    def _init_condition_data_schema(
        self,
        conditions: dict[str, Collection[str]] | None = None,
        conditions_reps: dict[str, str] | None = None,
        conditions_covariates: Collection[str] | None = None,
    ) -> ConditionDataSchema:
        """"""  # noqa
        return ConditionDataSchema(
            conditions={} if conditions is None else conditions,
            conditions_reps={} if conditions_reps is None else conditions_reps,
            conditions_covariates=[] if conditions_covariates is None else conditions_covariates,
        )

    def _init_target_data_schema(
        self,
        categorical_covs_dict: Mapping[str, TargetCovariatesEncodingId] | None = None,
        continuous_covs: Collection[str] | None = None,
    ) -> TargetDataSchema:
        """"""  # noqa
        return TargetDataSchema(
            categorical_covs_dict={} if categorical_covs_dict is None else categorical_covs_dict,
            continuous_covs=[] if continuous_covs is None else continuous_covs,
        )

    def _init_groups_data_schema(
        self,
        groups: Collection[str] | None = None,
        groups_reps: dict[str, str] | None = None,
        groups_encoding: dict[str, TargetCovariatesEncodingId] | None = None,
    ) -> GroupsDataSchema:
        """"""  # noqa
        return GroupsDataSchema(
            groups=[] if groups is None else groups,
            groups_reps={} if groups_reps is None else groups_reps,
            groups_encoding={} if groups_encoding is None else groups_encoding,
        )

    def _init_indexer(
        self,
        groups_cols: Collection[str] | None = None,
        conditions_cols: Collection[str] | None = None,
    ) -> HierarchicalIndexer:
        """"""  # noqa
        return HierarchicalIndexer(
            groups_cols=[] if groups_cols is None else groups_cols,  # TODO: add attributes for base groups
            conditions_cols=[] if conditions_cols is None else conditions_cols,
        )

    def _get_state_data(
        self,
        adata: AnnData,
    ) -> StateData:
        """"""  # noqa
        return self._state_data_schema.get_data(adata)

    def _get_condition_data(
        self,
        adata: AnnData,
    ) -> MixedTypeData:
        """"""  # noqa
        return self._condition_data_schema.get_data(adata)

    def _get_groups_data(
        self,
        adata: AnnData,
    ) -> CategoricalData:
        """"""  # noqa
        return self._groups_data_schema.get_data(adata)

    def _get_target_data(
        self,
        adata: AnnData,
    ) -> MixedTypeData:
        """"""  # noqa
        return self._target_data_schema.get_data(adata)

    def _get_compiled_data(
        self,
        adata: AnnData,
    ) -> DistributionData:
        """"""  # noqa

        # retrieving data
        state_data: StateData = self._get_state_data(adata)
        condition_data: MixedTypeData = self._get_condition_data(adata)
        target_data: MixedTypeData = self._get_target_data(adata)
        groups_data: CategoricalData = self._get_groups_data(adata)
        return DistributionData(
            state_data,
            target_data=target_data,
            condition_data=condition_data,
            groups_data=groups_data,
        )

    def get_matched_data(
        self,
        adata: AnnData,
    ) -> NestedData:
        """"""  # noqa
        data: DistributionData = self._get_compiled_data(adata)
        index: pd.DataFrame = self._indexer.create_index(data.ann_df)
        mapped_index: NestedMappedLevelIndex = self._selector.index_to_nested_dict(index)
        return NestedData.init_from_data(data, index, mapped_index, source_key=self.source_key)

    @property
    def control_values_dict(self) -> dict[str, str] | None:
        """"""  # noqa
        return self._control_values_dict

    @property
    def indexer(self) -> HierarchicalIndexer:
        """"""  # noqa
        return self._indexer

    @property
    def selector(self) -> IndexSelector:
        """"""  # noqa
        return self._selector

    @property
    def state_data_schema(self) -> StateDataSchema:
        """"""  # noqa
        return self._state_data_schema

    @property
    def condition_data_schema(self) -> ConditionDataSchema:
        """"""  # noqa
        return self._condition_data_schema

    @property
    def target_data_schema(self) -> ConditionDataSchema:
        """"""  # noqa
        return self._condition_data_schema

    @property
    def source_key(self) -> tuple[Any]:
        """"""  # noqa
        if self._control_values_dict is None:
            return None
        control_query_dict = {
            cond: self._control_values_dict[level]
            for level, conditions in self._condition_data_schema.conditions.items()
            for cond in conditions
        }
        return self._selector.query_factory.query_dict_to_tuple(control_query_dict)
