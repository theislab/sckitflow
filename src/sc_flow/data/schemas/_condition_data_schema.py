from collections.abc import Collection
from dataclasses import dataclass
from dataclasses import field as dc_field

import pandas as pd
from anndata import AnnData

from sc_flow._types import MappedArray
from sc_flow._utils import check_sequence_query_against_reference
from sc_flow.data._mixins import BatchMixin
from sc_flow.data._structures import CombinationData, ConditionData
from sc_flow.data.schemas._base_schema import BaseDataSchema

__all__ = ["ConditionDataSchema"]


@dataclass(frozen=True)
class ConditionDataSchema(BaseDataSchema):
    """Implements the logic for conditioning."""

    conditions: dict[str, Collection[str]] = dc_field(default_factory=lambda: {})
    conditions_reps: dict[str, str] = dc_field(default_factory=lambda: {})
    conditions_covariates: Collection[str] = dc_field(default_factory=lambda: [])

    def __post_init__(self) -> None:
        """"""  # noqa
        check_sequence_query_against_reference(
            self.conditions_reps.keys(),
            self.conditions.keys(),
            allow_missing_from_query=False,
            allow_missing_from_reference=False,
        )

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

    def _verify_categorical_covariates(
        self,
        adata: AnnData,
    ) -> None:
        """"""  # noqa
        for condition_realm, condition_rep in self.conditions_reps.items():
            if condition_realm not in self.conditions:
                msg = f"Condition {condition_realm} not found."
                raise ValueError(msg)
            self._check_key_found_in_adata_field(adata, condition_rep, "uns")

    def _verify_continuous_covariates(
        self,
        adata: AnnData,
    ) -> None:
        """"""  # noqa
        for covariate in self.conditions_covariates:
            self._check_key_found_in_adata_field(adata, covariate, "obsm")

    def _get_covariates_df(
        self,
        adata: AnnData,
    ) -> dict[str, pd.DataFrame]:
        """"""  # noqa
        return adata.obs.loc[:, self.all_condition_categories]

    def _get_repr_dict(self, adata: AnnData) -> dict[str, MappedArray]:
        """"""  # noqa
        return {
            condition_name: adata.uns[condition_repr] for condition_name, condition_repr in self.conditions_reps.items()
        }

    def _get_categorical_covariates(
        self,
        adata: AnnData,
    ) -> CombinationData:
        """"""  # noqa
        covariates_df = self._get_covariates_df(adata)
        repr_dict = self._get_repr_dict(adata)
        return CombinationData(covariates_df, repr_dict=repr_dict)

    def _get_continuous_covariates(
        self,
        adata: AnnData,
    ) -> BatchMixin:
        """"""  # noqa
        return {covariate_name: adata.obsm[covariate_name] for covariate_name in self.conditions_covariates}

    def _verify_schema(
        self,
        adata: AnnData,
    ) -> None:
        """"""  # noqa
        self._verify_categorical_covariates(adata)
        self._verify_continuous_covariates(adata)

    def _get_data(
        self,
        adata: AnnData,
    ) -> ConditionData:
        """"""  # noqa
        categorical_covariates = self._get_categorical_covariates(adata)
        continuous_covariates = self._get_continuous_covariates(adata)
        return ConditionData(condition_reps=categorical_covariates, condition_covariates=continuous_covariates)
