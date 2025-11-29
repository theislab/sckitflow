from collections.abc import Collection
from dataclasses import dataclass

from anndata import AnnData

from sc_flow.data._mixins import BatchMixin
from sc_flow.data._structures import CombinationData, ConditionData
from sc_flow.data.schemas._base_schema import BaseDataSchema

__all__ = ["ConditionDataSchema"]


@dataclass
class ConditionDataSchema(BaseDataSchema):
    """Implements the logic for conditioning."""

    conditions: dict[str, Collection[str]] | None = None
    conditions_reps: dict[str, str] | None = None
    conditions_covariates: Collection[str] | None = None

    def __post_init__(self) -> None:
        """"""  # noqa
        missing_keys = set(self.conditions.keys()) - set(self.conditions_reps.keys())
        if len(missing_keys):
            msg = "Keys of conditions and conditions_reps are shared"
            raise KeyError(msg)

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

    def _verify_schema_categorical_covariates(
        self,
        adata: AnnData,
    ) -> None:
        """"""  # noqa
        for condition_realm, condition_rep in self.conditions_reps.items():
            if condition_realm not in self.conditions:
                msg = f"Condition {condition_realm} not found."
                raise ValueError(msg)
            self._check_key_found_in_adata_field(adata, condition_rep, "uns")

    def _verify_schema_continuous_covariates(
        self,
        adata: AnnData,
    ) -> None:
        """"""  # noqa
        for condition_cat, covariates in self.conditions_covariates.items():
            if condition_cat not in self.all_condition_categories:
                msg = f"Condition category {condition_cat} not found."
                raise KeyError(msg)
            for covariate in covariates:
                self._check_key_found_in_adata_field(adata, covariate, "obsm")

    def _enforce_schema_categorical_covariates(
        self,
        adata: AnnData,
    ) -> CombinationData:
        """"""  # noqa
        column_values = ...
        repr_dict = ...
        raise NotImplementedError
        return CombinationData(
            column_values=column_values,
            repr_dict=repr_dict,
        )

    def _enforce_schema_continuous_covariates(
        self,
        adata: AnnData,
    ) -> BatchMixin:
        """"""  # noqa
        raise NotImplementedError

    def _verify_schema(
        self,
        adata: AnnData,
    ) -> None:
        """"""  # noqa
        self._verify_schema_categorical_covariates(adata)
        self._verify_schema_continuous_covariates(adata)

    def _enforce_schema(
        self,
        adata: AnnData,
    ) -> ConditionData:
        """"""  # noqa
        categorical_covariates = self._enforce_schema_categorical_covariates(adata)
        continuous_covariates = self._enforce_schema_continuous_covariates(adata)
        raise ConditionData(condition_reps=categorical_covariates, condition_covariates=continuous_covariates)
