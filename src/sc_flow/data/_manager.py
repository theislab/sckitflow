from collections.abc import Collection, Mapping
from dataclasses import dataclass

import pandas as pd
from anndata import AnnData

from sc_flow._types import TargetCovariatesEncoding
from sc_flow.data._indexer import HierarchicalIndexer
from sc_flow.data._structures import (
    ConditionData,
    IndexedContainer,
    StateData,
    TargetData,
)
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
        self._state_data_contract = StateDataSchema(
            sample_rep=self.sample_rep,
        )
        self._condition_data_contract = ConditionDataSchema(
            conditions_reps=self.conditions_reps, conditions_covariates=self.conditions_covariates
        )
        self._target_data_contract = TargetDataSchema(
            categorical_target_covariates=self.categorical_target_covariates,
            continuous_target_covariates=self.continuous_target_covariates,
        )
        self._indexer = HierarchicalIndexer(
            groups_cols=None,  # TODO: add attributes for base groups
            conditions_cols=self.condition_data_contract.all_condition_categories,
        )

    def _get_index(self, adata: AnnData) -> pd.MultiIndex:
        """"""  # noqa
        return self._indexer.create_index(adata.obs)

    def _get_state_data(
        self,
        adata: AnnData,
    ) -> StateData:
        """"""  # noqa
        return self.state_data_contract.enforce_contract(adata)

    def _get_condition_data(
        self,
        adata: AnnData,
    ) -> ConditionData:
        """"""  # noqa
        return self.condition_data_contract.enforce_contract(adata)

    def _get_target_data(
        self,
        adata: AnnData,
    ) -> TargetData:
        """"""  # noqa
        return self.target_data_contract.enforce_contract(adata)

    def _get_data(
        self,
        adata: AnnData,
    ) -> IndexedContainer:
        """"""  # noqa

        # retrieving data
        state_data = self._get_state_data(adata)
        condition_data = self._get_condition_data(adata)
        target_data = self._get_target_data(adata)

        return IndexedContainer(
            state_data,
            target_data=target_data,
            condition_data=condition_data,
        )

    def _get_index(
        self,
        adata: AnnData,
    ) -> pd.MultiIndex:
        """"""  # noqa
        return self._indexer.create_index(adata.obs)

    @property
    def indexer(self) -> HierarchicalIndexer:
        """"""  # noqa
        return self._indexer

    @property
    def state_data_contract(self) -> StateDataSchema:
        """"""  # noqa
        return self._state_data_contract

    @property
    def condition_data_contract(self) -> ConditionDataSchema:
        """"""  # noqa
        return self._condition_data_contract

    @property
    def target_data_contract(self) -> ConditionDataSchema:
        """"""  # noqa
        return self._condition_data_contract
