from collections.abc import Mapping

from anndata import AnnData

from sc_flow.core.data._encoders import Encoder, Lookup
from sc_flow.core.data.schemas._base_schema import StrictDataSchema

__all__ = ["CovariatesDataSchema"]


class CovariatesDataSchema(StrictDataSchema):
    """Data schema for embedded per-sample covariates (schema-generalization Change 1).

    This is the *embedding* axis only — it feeds the conditioner, never the matching split (that is
    ``compile_obs(..., match_context=...)``). It replaces the old ``GroupsDataSchema``, whose ``groups``
    list did double duty (split *and* embed). The embedded set is **derived** from the encoder map's
    keys — there is no separate column list to keep in sync.

    Example::

        >>> from sc_flow.core.data._encoders import lookup
        >>> CovariatesDataSchema(covariate_encoders={"cell_type": lookup("cell_type")})

    :param covariate_encoders: Mapping ``{column: Encoder}`` — one encoder per embedded covariate.
    """

    def __init__(
        self,
        covariate_encoders: Mapping[str, Encoder] | None = None,
    ) -> None:
        self._covariate_encoders = {} if covariate_encoders is None else covariate_encoders
        super().__init__()

    def _verify_args(self) -> None:
        """No structural argument checks beyond the encoder map being present."""

    def _verify_schema(self, adata: AnnData) -> None:
        """Every covariate column must exist in ``obs``; every lookup encoder's table must be in ``uns``."""
        for col, encoder in self._covariate_encoders.items():
            self._check_key_found_in_adata_field(adata, col, "obs")
            if isinstance(encoder, Lookup):
                self._check_key_found_in_adata_field(adata, encoder.uns_key, "uns")

    @property
    def covariates(self) -> list[str]:
        """The embedded covariate columns — derived from the encoder map's keys."""
        return list(self._covariate_encoders)

    @property
    def covariate_encoders(self) -> Mapping[str, Encoder]:
        """Per-covariate encoder map (``lookup``/``one_hot``/``label``/``functional``)."""
        return self._covariate_encoders

    @property
    def categorical_reps_map(self) -> dict[str, str]:
        """Each covariate is its own realm."""
        return {covariate: covariate for covariate in self.covariates}
