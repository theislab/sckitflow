from collections.abc import Collection
from dataclasses import dataclass

from anndata import AnnData

from sc_flow.data._data_structures import CombinatorialCategoricalDataContainer, ConditionDataContainer
from sc_flow.data._mixins import BatchMixin
from sc_flow.data.contracts._base_contract import BaseDataContract

__all__ = ["ConditionDataContract"]


@dataclass
class ConditionDataContract(BaseDataContract):
    """Implements the logic for conditioning."""

    conditions: dict[str, Collection[str]] | None = None
    conditions_reps: dict[str, str] | None = None
    conditions_covariates: Collection[str] | None = None

    @property
    def all_condition_categories(
        self,
    ) -> Collection[str]:
        """"""  # noqa
        if self.conditions is None:
            return ()
        return tuple(cat for condition in self.conditions.values() for cat in condition)

    @property
    def condition_category_to_realm(
        self,
    ) -> dict[str, str]:
        """"""  # noqa
        cat2realm = {}
        for condition, condition_cats in self.conditions.items():
            for cat in condition_cats:
                cat2realm[cat] = condition
        return cat2realm

    @property
    def allows_ot_coupling(
        self,
    ) -> bool:
        """"""  # noqa
        return self.conditions_covariates is None

    def _verify_contract_categorical_covariates(
        self,
        adata: AnnData,
    ) -> None:
        """"""  # noqa
        raise NotImplementedError

    def _verify_contract_continuous_covariates(
        self,
        adata: AnnData,
    ) -> None:
        """"""  # noqa
        raise NotImplementedError

    def _enforce_contract_categorical_covariates(
        self,
        adata: AnnData,
    ) -> CombinatorialCategoricalDataContainer:
        """"""  # noqa
        raise NotImplementedError

    def _enforce_contract_continuous_covariates(
        self,
        adata: AnnData,
    ) -> BatchMixin:
        """"""  # noqa
        raise NotImplementedError

    def _verify_contract(
        self,
        adata: AnnData,
    ) -> None:
        """"""  # noqa
        self._verify_contract_categorical_covariates(adata)
        self._verify_contract_continuous_covariates(adata)

    def _enforce_contract(
        self,
        adata: AnnData,
    ) -> ConditionDataContainer:
        """"""  # noqa
        categorical_covariates = self._enforce_contract_categorical_covariates(adata)  # noqa
        continuous_covariates = self._enforce_contract_continuous_covariates(adata)  # noqa
        raise NotImplementedError
