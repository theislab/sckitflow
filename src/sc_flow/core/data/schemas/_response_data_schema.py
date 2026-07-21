from collections.abc import Collection, Mapping

from anndata import AnnData

from sc_flow.core.data._encoders import Encoder, Lookup
from sc_flow.core.data.schemas._base_schema import StrictDataSchema

__all__ = ["ResponseDataSchema"]


class ResponseDataSchema(StrictDataSchema):
    """Data schema for response data — categorical covariates (one encoder each) plus continuous ones.

    After Change 2 the categorical covariates are declared with a single ``{column: Encoder}`` map
    (:param:`response_encoders`), the same abstraction used by conditions and covariates.

    :param response_encoders: Mapping ``{column: Encoder}`` — one encoder per categorical response covariate.
    :param continuous_covs: Continuous response covariates (keys in ``.obsm``).
    """

    def __init__(
        self,
        response_encoders: Mapping[str, Encoder] | None = None,
        continuous_covs: Collection[str] | None = None,
    ) -> None:
        self._response_encoders = {} if response_encoders is None else response_encoders
        self._continuous_covs = [] if continuous_covs is None else continuous_covs
        super().__init__()

    def _verify_args(self) -> None:
        """No structural argument checks beyond the encoder map being present."""

    def _verify_schema(self, adata: AnnData) -> None:
        """Categorical columns must be in ``obs`` (lookups' tables in ``uns``); continuous in ``obsm``."""
        for col, encoder in self._response_encoders.items():
            self._check_key_found_in_adata_field(adata, col, "obs")
            if isinstance(encoder, Lookup):
                self._check_key_found_in_adata_field(adata, encoder.uns_key, "uns")
        for covariate in self._continuous_covs:
            self._check_key_found_in_adata_field(adata, covariate, "obsm")

    @property
    def categorical_covariates(self) -> tuple[str]:
        """The categorical response covariate columns — the encoder map's keys."""
        return tuple(self._response_encoders.keys())

    @property
    def response_encoders(self) -> Mapping[str, Encoder]:
        """Per-covariate encoder map (``lookup``/``one_hot``/``label``/``functional``)."""
        return self._response_encoders

    @property
    def continuous_covs(self) -> Collection[str]:
        """Exposes to `continuous_covs` parameter set at initialization."""
        return self._continuous_covs

    @property
    def has_categorical_covariates(self) -> bool:
        """Whether the response schema includes categorical covariates."""
        return len(self._response_encoders) > 0

    @property
    def has_continuous_covariates(self) -> bool:
        """Whether the response schema includes continuous covariates."""
        return len(self._continuous_covs) > 0

    @property
    def categorical_reps_map(self) -> dict[str, str]:
        """Each categorical response covariate is its own realm."""
        return {covariate: covariate for covariate in self._response_encoders}
