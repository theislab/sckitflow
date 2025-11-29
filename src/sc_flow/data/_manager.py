from collections.abc import Collection, Mapping
from dataclasses import dataclass

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


@dataclass
class DataManager:
    """"""  # noqa

    sample_rep: str | None = None
    conditions: dict[str, Collection[str]] | None = None
    conditions_reps: dict[str, str] | None = None
    conditions_covariates: Collection[str] | None = None
    categorical_target_covariates: Mapping[str, TargetCovariatesEncoding] | None = None
    continuous_target_covariates: Collection[str] | None = None

    def __post_init__(self) -> None:
        """"""  # noqa
        self._state_data_schema = StateDataSchema(
            sample_rep=self.sample_rep,
        )
        self._condition_data_schema = ConditionDataSchema(
            conditions_reps=self.conditions_reps, conditions_covariates=self.conditions_covariates
        )
        self._target_data_schema = TargetDataSchema(
            categorical_target_covariates=self.categorical_target_covariates,
            continuous_target_covariates=self.continuous_target_covariates,
        )
        self._indexer = HierarchicalIndexer(
            groups_cols=None,  # TODO: add attributes for base groups
            conditions_cols=self.condition_data_schema.all_condition_categories,
        )
        self._selector = IndexSelector.init_from_indexer(self.indexer)

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
