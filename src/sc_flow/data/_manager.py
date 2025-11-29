from collections.abc import Collection, Mapping

import pandas as pd
from anndata import AnnData

from sc_flow._types import TargetCovariatesEncoding
from sc_flow.data._collections import TrainCollection
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
        categorical_covs_dict: Mapping[str, TargetCovariatesEncoding] | None = None,
        continuous_covs_dict: Collection[str] | None = None,
    ) -> None:
        """"""  # noqa

        self._sample_rep = sample_rep
        self._conditions = conditions
        self._conditions_reps = conditions_reps
        self._conditions_covariates = conditions_covariates
        self._categorical_covs_dict = categorical_covs_dict
        self._continuous_covs_dict = continuous_covs_dict

        self._state_data_schema: StateDataSchema | None = None
        self._condition_data_schema: ConditionDataSchema | None = None
        self._target_data_schema: TargetDataSchema | None = None
        self._indexer: HierarchicalIndexer | None = None
        self._selector: IndexSelector | None = None

    def _prepare_schemas(self) -> None:
        """"""  # noqa
        self._state_data_schema = self._init_state_data_schema()
        self._condition_data_schema = self._init_condition_data_schema()
        self._target_data_schema = self._init_target_data_schema()
        self._indexer = self._init_indexer()
        self._selector = IndexSelector.init_from_indexer(self.indexer)

    def _init_state_data_schema(self) -> StateDataSchema:
        """"""  # noqa
        return StateDataSchema(sample_rep=self._sample_rep)

    def _init_condition_data_schema(self) -> ConditionDataSchema:
        """"""  # noqa
        return ConditionDataSchema(
            conditions_reps=self._conditions_reps, conditions_covariates=self._conditions_covariates
        )

    def _init_target_data_schema(self) -> TargetDataSchema:
        """"""  # noqa
        return TargetDataSchema(
            categorical_covs_dict=self._categorical_covs_dict,
            continuous_covs_dict=self._continuous_covs_dict,
        )

    def _init_indexer(self) -> HierarchicalIndexer:
        """"""  # noqa
        return HierarchicalIndexer(
            groups_cols=None,  # TODO: add attributes for base groups
            conditions_cols=self.condition_data_schema.all_condition_categories,
        )

    def _get_index(self, adata: AnnData) -> pd.MultiIndex:
        """"""  # noqa
        return self._indexer.create_index(adata.obs)

    def _get_state_data(
        self,
        adata: AnnData,
    ) -> StateData:
        """"""  # noqa
        return self.state_data_schema.enforce_schema(adata)

    def _get_condition_data(
        self,
        adata: AnnData,
    ) -> ConditionData:
        """"""  # noqa
        return self.condition_data_schema.enforce_schema(adata)

    def _get_target_data(
        self,
        adata: AnnData,
    ) -> TargetData:
        """"""  # noqa
        return self.target_data_schema.enforce_schema(adata)

    def _get_compiled_data(
        self,
        adata: AnnData,
    ) -> CompiledData:
        """"""  # noqa

        # retrieving data
        state_data = self._get_state_data(adata)
        condition_data = self._get_condition_data(adata)
        target_data = self._get_target_data(adata)

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
