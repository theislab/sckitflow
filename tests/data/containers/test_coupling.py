import numpy as np
import pytest
from anndata import AnnData

from sc_flow.data.containers import CouplingData, StateData


class TestCouplingData:
    @pytest.mark.parametrize("n_shared_dims", [None, 10, 500])
    def test_init_from_state_data(self, adata: AnnData, n_shared_dims: int | None) -> None:
        X = adata.X
        state_data = StateData(X)
        if n_shared_dims is not None and n_shared_dims > X.shape[1]:
            with pytest.raises(ValueError):
                coupling_data = CouplingData.init_from_state_data(state_data, n_shared_dims=n_shared_dims)
            return None
        coupling_data = CouplingData.init_from_state_data(state_data, n_shared_dims=n_shared_dims)
        assert isinstance(coupling_data.state_lin, StateData)
        if n_shared_dims is None:
            assert coupling_data.state_quad is None
            assert coupling_data.state_lin.X.shape[1] == X.shape[1]
            joint_arr = coupling_data.state_lin.X
        else:
            assert isinstance(coupling_data.state_quad, StateData)
            assert coupling_data.state_lin.X.shape[1] + coupling_data.state_quad.X.shape[1] == X.shape[1]
            joint_arr = np.concatenate((coupling_data.state_lin.X, coupling_data.state_quad.X), axis=1)
        assert np.all(joint_arr == X)

    @pytest.mark.parametrize("n_shared_dims", [None, 10])
    def test_assert_same_spatial_dims(
        self,
        adata: AnnData,
        n_shared_dims: int | None,
    ) -> None:
        X = adata.X
        state_data = StateData(X)

        ref_coupling = CouplingData.init_from_state_data(state_data, n_shared_dims=None)
        tgt_coupling = CouplingData.init_from_state_data(state_data, n_shared_dims=n_shared_dims)
        if n_shared_dims is None:
            ref_coupling.assert_same_spatial_dims(tgt_coupling)
        else:
            with pytest.raises(ValueError):
                ref_coupling.assert_same_spatial_dims(tgt_coupling)
