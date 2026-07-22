from anndata import AnnData

from sc_flow.data.schemas._base_schema import StrictDataSchema

__all__ = ["StateDataSchema"]


class StateDataSchema(StrictDataSchema):

    def __init__(
        self,
        sample_rep: str | None = None,
    ) -> None:
        self._sample_rep = sample_rep
        super().__init__()

    def _verify_args(self) -> None:
        pass

    def _verify_schema(
        self,
        adata: AnnData,
    ) -> None:
        # when we provide the sample rep key it should appear in `self.adata.obsm`
        if self._sample_rep is not None:
            self._check_key_found_in_adata_field(adata, self._sample_rep, "obsm")

    @property
    def sample_rep(self) -> str:
        return self._sample_rep
