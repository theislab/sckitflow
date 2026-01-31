from anndata import AnnData

from sc_flow.data.containers._state import StateData
from sc_flow.data.schemas._base_schema import StrictDataSchema

__all__ = ["StateDataSchema"]


class StateDataSchema(StrictDataSchema):
    """Data Schema Implementing the logic for state data."""

    def __init__(
        self,
        sample_rep: str | None = None,
    ) -> None:
        """Initializes the state data schema

        :param:
        """
        self._sample_rep = sample_rep
        super().__init__()

    def _verify_args(self) -> None:
        """No-op needed for compatibility with parent class."""
        pass

    def _verify_schema(
        self,
        adata: AnnData,
    ) -> None:
        """Verifies the state representation settings on the input :class: `AnnData`.

        :param adata: The input annotated data to verify.
        :type adata: class: `AnnData`
        """
        # when we provide the sample rep key it should appear in `self.adata.obsm`
        if self._sample_rep is not None:
            self._check_key_found_in_adata_field(adata, self._sample_rep, "obsm")

    def _get_data(
        self,
        adata: AnnData,
    ) -> StateData:
        """Enforces the schema on the input :class: `AnnData`.

        :param adata: The input data.
        :type adata: class: `AnnData`
        """
        X = self._extract_array(adata, self._sample_rep)
        return StateData(X)

    @property
    def sample_rep(self) -> str:
        """Exposes to `sample_rep` parameter set at initialization."""
        return self._sample_rep
