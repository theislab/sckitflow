import pytest
from anndata import AnnData

from sc_flow.data._structures import StateData
from sc_flow.data.schemas import StateDataSchema

inval_key: str = "invalid_key"


class TestConditionDataSchema:
    @pytest.mark.parametrize("sample_rep", [None, "X_repr", inval_key])
    def test_init(
        self,
        n_obs: int,
        n_feats_obsm_repr: int,
        n_genes: int,
        adata: AnnData,
        sample_rep: str | None,
    ) -> None:
        """"""

        schema = StateDataSchema(sample_rep=sample_rep)
        if sample_rep == inval_key:
            with pytest.raises(KeyError, match=f"Key '{sample_rep}' not found in adata.obsm"):
                data = schema.get_data(adata)
            return None
        data = schema.get_data(adata)
        assert isinstance(data, StateData)
        if sample_rep == "X_repr":
            target_shape = (n_obs, n_feats_obsm_repr)
        else:
            target_shape = (n_obs, n_genes)
        assert data.X.shape == target_shape
