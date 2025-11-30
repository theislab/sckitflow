from dataclasses import dataclass

from anndata import AnnData

from sc_flow.data._structures import StateData
from sc_flow.data.schemas._base_schema import BaseDataSchema

__all__ = ["StateDataSchema"]


@dataclass(frozen=True)
class StateDataSchema(BaseDataSchema):
    """"""  # noqa

    sample_rep: str | None = None

    def _verify_schema(
        self,
        adata: AnnData,
    ) -> None:
        """Verifies the state representation settings on the input :class: `AnnData`.

        :param adata: The input annotated data to verify.
        :type adata: class: `AnnData`
        """
        # when we provide the sample rep key it should appear in `self.adata.obsm`
        if self.sample_rep is not None:
            self._check_key_found_in_adata_field(adata, self.sample_rep, "obsm")

    def _get_data(
        self,
        adata: AnnData,
    ) -> StateData:
        """"""  # noqa
        if self.sample_rep is None:
            X = adata.X
        X = adata.obsm[self.sample_rep]
        return StateData(X)
