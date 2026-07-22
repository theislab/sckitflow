from collections.abc import Collection, Mapping

from anndata import AnnData

from scfit._utils import check_sequence_query_against_reference
from sc_flow.data._encoders import Encoder, Lookup
from sc_flow.data.schemas._base_schema import StrictDataSchema

__all__ = ["ConditionDataSchema"]


class ConditionDataSchema(StrictDataSchema):

    def __init__(
        self,
        conditions: dict[str, Collection[str]] | None = None,
        condition_encoders: Mapping[str, Encoder] | None = None,
        conditions_covariates: Collection[str] | None = None,
    ) -> None:
        self._conditions = {} if conditions is None else conditions
        self._condition_encoders = {} if condition_encoders is None else condition_encoders
        self._conditions_covariates = () if conditions_covariates is None else conditions_covariates
        super().__init__()

    def _verify_args(self) -> None:
        check_sequence_query_against_reference(
            self._conditions.keys(),
            self._condition_encoders.keys(),
            allow_missing_from_query=False,
            allow_missing_from_reference=False,
        )

    def _verify_categorical_covariates(self, adata: AnnData) -> None:
        for condition_cols in self._conditions.values():
            for col in condition_cols:
                self._check_key_found_in_adata_field(adata, col, "obs")
        for encoder in self._condition_encoders.values():
            if isinstance(encoder, Lookup):
                self._check_key_found_in_adata_field(adata, encoder.uns_key, "uns")

    def _verify_continuous_covariates(self, adata: AnnData) -> None:
        for covariate in self._conditions_covariates:
            self._check_key_found_in_adata_field(adata, covariate, "obsm")

    def _verify_schema(self, adata: AnnData) -> None:
        self._verify_categorical_covariates(adata)
        self._verify_continuous_covariates(adata)

    @property
    def all_condition_cols(self) -> tuple[str]:
        return tuple(cat for condition in self._conditions.values() for cat in condition)

    @property
    def condition_col_to_level(self) -> dict[str, str]:
        col2level = {}
        for condition, condition_cols in self._conditions.items():
            for cat in condition_cols:
                col2level[cat] = condition
        return col2level

    @property
    def allows_grouping(self) -> bool:
        return len(self._conditions_covariates) == 0

    @property
    def conditions(self) -> dict[str, Collection[str]]:
        return self._conditions

    @property
    def condition_encoders(self) -> Mapping[str, Encoder]:
        return self._condition_encoders

    @property
    def conditions_covariates(self) -> Collection[str]:
        return self._conditions_covariates

    @property
    def has_categorical_covariates(self) -> bool:
        return len(self._conditions) > 0

    @property
    def has_continuous_covariates(self) -> bool:
        return len(self._conditions_covariates) > 0

    @property
    def categorical_reps_map(self) -> dict[str, str]:
        reps_map = {}
        for realm, cov_list in self.conditions.items():
            for cov in cov_list:
                reps_map[cov] = realm
        return reps_map
