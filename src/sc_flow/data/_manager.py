from collections.abc import Collection, Mapping

import pandas as pd
from anndata import AnnData

from sc_flow._types import TargetCovariatesEncoding
from sc_flow.data._collections import TrainCollection, ValidationCollection
from sc_flow.data._structures import (
    CompiledData,
    ConditionData,
    StateData,
    TargetData,
)
from sc_flow.data.grouping._indexer import HierarchicalIndexer
from sc_flow.data.grouping._selector import IndexSelector
from sc_flow.data.schemas import ConditionDataSchema, StateDataSchema, TargetDataSchema

__all__ = ["DataManager"]


class DataManager:
    """"""  # noqa

    def __init__(
        self,
        sample_rep: str | None = None,
        conditions: dict[str, Collection[str]] | None = None,
        conditions_reps: dict[str, str] | None = None,
        conditions_covariates: Collection[str] | None = None,
        target_categorical_covs_dict: Mapping[str, TargetCovariatesEncoding] | None = None,
        target_continuous_covs_dict: Collection[str] | None = None,
        groups: dict[str, Collection[str]] | None = None,
        groups_reps: dict[str, str] | None = None,
        groups_encoding: dict[str, TargetCovariatesEncoding | None] | None = None,
    ) -> None:
        """"""  # noqa

        self._state_data_schema: StateDataSchema = self._init_state_data_schema(sample_rep=sample_rep)
        self._condition_data_schema: ConditionDataSchema = self._init_condition_data_schema(
            conditions=conditions,
            conditions_reps=conditions_reps,
            conditions_covariates=conditions_covariates,
        )
        self._target_data_schema: TargetDataSchema = self._init_target_data_schema(
            categorical_covs_dict=target_categorical_covs_dict,
            continuous_covs_dict=target_continuous_covs_dict,
        )
        self._group_data = self._init_groups_data(
            groups=groups,
            groups_reps=groups_reps,
            groups_encoding=groups_encoding,
        )
        self._indexer: HierarchicalIndexer = self._init_indexer(
            groups_cols=None, conditions_cols=self._condition_data_schema.all_condition_categories
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
            conditions=conditions,
            conditions_reps=conditions_reps,
            conditions_covariates=conditions_covariates,
        )

    def _init_target_data_schema(
        self,
        target_categorical_covs_dict: Mapping[str, TargetCovariatesEncoding] | None = None,
        target_continuous_covs_dict: Collection[str] | None = None,
    ) -> TargetDataSchema:
        """"""  # noqa
        return TargetDataSchema(
            categorical_covs_dict=target_categorical_covs_dict,
            continuous_covs_dict=target_continuous_covs_dict,
        )

    def _init_groups_data(
        self,
        groups: dict[str, Collection[str]] | None = None,
        groups_reps: dict[str, str] | None = None,
        groups_encoding: dict[str, TargetCovariatesEncoding | None] | None = None,
    ) -> None:
        """"""  # noqa
        raise NotImplementedError

    def _init_indexer(
        self,
        groups_cols: Collection[str] | None = None,
        conditions_cols: Collection[str] | None = None,
    ) -> HierarchicalIndexer:
        """"""  # noqa
        return HierarchicalIndexer(
            groups_cols=groups_cols,  # TODO: add attributes for base groups
            conditions_cols=conditions_cols,
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
    ) -> ConditionData:
        """"""  # noqa
        return self._condition_data_schema.get_data(adata)

    def _get_target_data(
        self,
        adata: AnnData,
    ) -> TargetData:
        """"""  # noqa
        return self._target_data_schema.get_data(adata)

    def _get_compiled_data(
        self,
        adata: AnnData,
    ) -> CompiledData:
        """"""  # noqa

        # retrieving data
        state_data: StateData = self._get_state_data(adata)
        condition_data: ConditionData = self._get_condition_data(adata)
        target_data: TargetData = self._get_target_data(adata)

        return CompiledData(
            state_data,
            target_data=target_data,
            condition_data=condition_data,
        )

    def get_train_collection(
        self,
        adata: AnnData,
    ) -> pd.MultiIndex:
        """"""  # noqa
        compiled_data = self._get_compiled_data(adata)
        return TrainCollection(
            compiled_data,
            self.indexer,
            self.selector,
        )

    def get_val_collection(
        self,
        adata: AnnData,
    ) -> pd.MultiIndex:
        """"""  # noqa
        compiled_data = self._get_compiled_data(adata)
        return ValidationCollection(
            compiled_data,
            self.indexer,
            self.selector,
        )

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
