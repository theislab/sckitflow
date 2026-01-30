import numpy as np
import pytest
from anndata import AnnData

from sc_flow.data.containers import CouplingData, StateData


class TestCouplingData:
    # -----------------------
    # init_from_state_data
    # -----------------------

    @pytest.mark.parametrize("n_shared_dims", [None, 1, 5])
    def test_init_from_state_data_valid(
        self,
        adata: AnnData,
        n_shared_dims: int | None,
    ) -> None:
        X = adata.X
        state_data = StateData(X)

        coupling = CouplingData.init_from_state_data(
            state_data,
            n_shared_dims=n_shared_dims,
        )

        assert isinstance(coupling, CouplingData)
        assert isinstance(coupling.state_lin, StateData)
        assert len(coupling) == X.shape[0]

        if n_shared_dims is None:
            assert coupling.state_quad is None
            assert coupling.state_lin.X.shape == X.shape
            np.testing.assert_array_equal(coupling.state_lin.X, X)
        else:
            assert isinstance(coupling.state_quad, StateData)
            assert coupling.state_lin.X.shape[1] == n_shared_dims
            assert coupling.state_quad.X.shape[1] == X.shape[1] - n_shared_dims

            reconstructed = np.concatenate(
                (coupling.state_lin.X, coupling.state_quad.X),
                axis=1,
            )
            np.testing.assert_array_equal(reconstructed, X)

    @pytest.mark.parametrize("n_shared_dims", [0, 9999])
    def test_init_from_state_data_invalid_n_shared_dims(
        self,
        adata: AnnData,
        n_shared_dims: int,
    ) -> None:
        X = adata.X
        state_data = StateData(X)

        with pytest.raises(ValueError, match="number of shared spatial dimensions"):
            CouplingData.init_from_state_data(
                state_data,
                n_shared_dims=n_shared_dims,
            )

    # -----------------------
    # __post_init__
    # -----------------------

    def test_post_init_mismatched_n_obs(self, adata: AnnData) -> None:
        X = adata.X
        state_lin = StateData(X)
        state_quad = StateData(X[:-1])

        with pytest.raises(ValueError):
            CouplingData(state_lin=state_lin, state_quad=state_quad)

    # -----------------------
    # __len__
    # -----------------------

    def test_len(self, adata: AnnData) -> None:
        coupling = CouplingData.init_from_state_data(StateData(adata.X))
        assert len(coupling) == adata.n_obs

    # -----------------------
    # __getitem__
    # -----------------------

    @pytest.mark.parametrize("idxs", [slice(0, 5), np.array([0, 2, 4])])
    def test_getitem(self, adata: AnnData, idxs) -> None:
        coupling = CouplingData.init_from_state_data(
            StateData(adata.X),
            n_shared_dims=3,
        )

        subset = coupling[idxs]

        assert isinstance(subset, CouplingData)
        assert len(subset) == len(coupling.state_lin[idxs])
        assert subset.state_lin.X.shape[1] == coupling.state_lin.X.shape[1]
        assert subset.state_quad.X.shape[1] == coupling.state_quad.X.shape[1]

    # -----------------------
    # assert_same_spatial_dims
    # -----------------------

    def test_assert_same_spatial_dims_valid(self, adata: AnnData) -> None:
        state_data = StateData(adata.X)

        c1 = CouplingData.init_from_state_data(state_data, n_shared_dims=3)
        c2 = CouplingData.init_from_state_data(state_data, n_shared_dims=3)

        c1.assert_same_spatial_dims(c2)

    def test_assert_same_spatial_dims_invalid(self, adata: AnnData) -> None:
        state_data = StateData(adata.X)

        c1 = CouplingData.init_from_state_data(state_data, n_shared_dims=2)
        c2 = CouplingData.init_from_state_data(state_data, n_shared_dims=5)

        with pytest.raises(ValueError, match="same number of spatial dimensions"):
            c1.assert_same_spatial_dims(c2)

    # -----------------------
    # __repr__
    # -----------------------

    def test_repr(self, adata: AnnData) -> None:
        coupling = CouplingData.init_from_state_data(
            StateData(adata.X),
            n_shared_dims=3,
        )

        rep = repr(coupling)

        assert "CouplingData" in rep
        assert "linear" in rep
        assert "quadratic" in rep
        assert f"n_obs={adata.n_obs}" in rep
