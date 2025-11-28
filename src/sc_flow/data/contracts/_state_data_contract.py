from dataclasses import dataclass

from anndata import AnnData

from sc_flow.data._data_structures import StateDataContainer
from sc_flow.data.contracts._base_contract import BaseDataContract

__all__ = ["StateDataContract"]


@dataclass
class StateDataContract(BaseDataContract):
    """"""  # noqa

    sample_rep: str | None = None

    def _verify_contract(
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

    def _enforce_contract(
        self,
        adata: AnnData,
    ) -> StateDataContainer:
        """"""  # noqa
        if self.sample_rep is None:
            X = adata.X
        X = adata.obsm[self.sample_rep]
        return StateDataContainer(X)
