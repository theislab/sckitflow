from collections.abc import Collection, Mapping

from anndata import AnnData

from sc_flow.data._encoders import Encoder, Lookup
from sc_flow.data.schemas._base_schema import StrictDataSchema

__all__ = ["ResponseDataSchema"]


class ResponseDataSchema(StrictDataSchema):

    def __init__(
        self,
        response_encoders: Mapping[str, Encoder] | None = None,
        continuous_covs: Collection[str] | None = None,
    ) -> None:
        self._response_encoders = {} if response_encoders is None else response_encoders
        self._continuous_covs = [] if continuous_covs is None else continuous_covs
        super().__init__()

    def _verify_args(self) -> None:
        ...

    def _verify_schema(self, adata: AnnData) -> None:
        for col, encoder in self._response_encoders.items():
            self._check_key_found_in_adata_field(adata, col, "obs")
            if isinstance(encoder, Lookup):
                self._check_key_found_in_adata_field(adata, encoder.uns_key, "uns")
        for covariate in self._continuous_covs:
            self._check_key_found_in_adata_field(adata, covariate, "obsm")

    @property
    def categorical_covariates(self) -> tuple[str]:
        return tuple(self._response_encoders.keys())

    @property
    def response_encoders(self) -> Mapping[str, Encoder]:
        return self._response_encoders

    @property
    def continuous_covs(self) -> Collection[str]:
        return self._continuous_covs

    @property
    def has_categorical_covariates(self) -> bool:
        return len(self._response_encoders) > 0

    @property
    def has_continuous_covariates(self) -> bool:
        return len(self._continuous_covs) > 0

    @property
    def categorical_reps_map(self) -> dict[str, str]:
        return {covariate: covariate for covariate in self._response_encoders}
