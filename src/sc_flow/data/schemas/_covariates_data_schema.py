from collections.abc import Mapping

from anndata import AnnData

from sc_flow.data._encoders import Encoder, Lookup
from sc_flow.data.schemas._base_schema import StrictDataSchema

__all__ = ["CovariatesDataSchema"]


class CovariatesDataSchema(StrictDataSchema):

    def __init__(
        self,
        covariate_encoders: Mapping[str, Encoder] | None = None,
    ) -> None:
        self._covariate_encoders = {} if covariate_encoders is None else covariate_encoders
        super().__init__()

    def _verify_args(self) -> None:
        ...

    def _verify_schema(self, adata: AnnData) -> None:
        for col, encoder in self._covariate_encoders.items():
            self._check_key_found_in_adata_field(adata, col, "obs")
            if isinstance(encoder, Lookup):
                self._check_key_found_in_adata_field(adata, encoder.uns_key, "uns")

    @property
    def covariates(self) -> list[str]:
        return list(self._covariate_encoders)

    @property
    def covariate_encoders(self) -> Mapping[str, Encoder]:
        return self._covariate_encoders

    @property
    def categorical_reps_map(self) -> dict[str, str]:
        return {covariate: covariate for covariate in self.covariates}
