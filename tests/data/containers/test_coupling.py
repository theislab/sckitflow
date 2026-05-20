import numpy as np
import pytest
from anndata import AnnData

from sc_flow.data.containers import CouplingData, StateData


class TestCouplingData:
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

    def test_post_init_mismatched_n_obs(self, adata: AnnData) -> None:
        X = adata.X
        state_lin = StateData(X)
        state_quad = StateData(X[:-1])

        with pytest.raises(ValueError):
            CouplingData(state_lin=state_lin, state_quad=state_quad)

    def test_len(self, adata: AnnData) -> None:
        coupling = CouplingData.init_from_state_data(StateData(adata.X))
        assert len(coupling) == adata.n_obs

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

    def test_concat_collection_two_full_couplings(self, adata: AnnData) -> None:
        """Concatenate two CouplingData objects that both have linear and quadratic parts."""
        state_data = StateData(adata.X)
        c1 = CouplingData.init_from_state_data(state_data, n_shared_dims=3)
        c2 = CouplingData.init_from_state_data(state_data, n_shared_dims=3)

        concatenated = CouplingData.concat_collection([c1, c2])

        assert len(concatenated) == len(c1) + len(c2)
        # Linear part should be vertically stacked
        expected_lin = np.vstack([c1.state_lin.X, c2.state_lin.X])
        np.testing.assert_array_equal(concatenated.state_lin.X, expected_lin)
        # Quadratic part likewise
        expected_quad = np.vstack([c1.state_quad.X, c2.state_quad.X])
        np.testing.assert_array_equal(concatenated.state_quad.X, expected_quad)

    def test_concat_collection_mixed_linear_only(self, adata: AnnData) -> None:
        """Concatenate objects where some have only linear term (quad=None)."""
        full_state = StateData(adata.X)
        c_full = CouplingData.init_from_state_data(full_state, n_shared_dims=2)  # has quad

        # Create a CouplingData with only linear term (no quadratic)
        linear_only = CouplingData(state_lin=StateData(adata.X[:5]), state_quad=None)

        # First element has quad, second does not -> should raise
        with pytest.raises(ValueError, match="incompatible data"):
            CouplingData.concat_collection([c_full, linear_only])

        # Reverse order: first element has no quad, second has -> also raise
        with pytest.raises(ValueError, match="incompatible data"):
            CouplingData.concat_collection([linear_only, c_full])

        # Both have only linear (quad=None) – should work
        c_lin1 = CouplingData(state_lin=StateData(adata.X[:10]), state_quad=None)
        c_lin2 = CouplingData(state_lin=StateData(adata.X[10:20]), state_quad=None)
        concat_lin = CouplingData.concat_collection([c_lin1, c_lin2])
        assert concat_lin.state_quad is None
        expected_lin = np.vstack([c_lin1.state_lin.X, c_lin2.state_lin.X])
        np.testing.assert_array_equal(concat_lin.state_lin.X, expected_lin)

    def test_concat_collection_mixed_quadratic_only(self, adata: AnnData) -> None:
        """Test objects that have only quadratic term (lin=None). Note: __post_init__ requires at least one not None."""
        # Create two objects with only quadratic term
        q1 = CouplingData(state_lin=None, state_quad=StateData(adata.X[:5]))
        q2 = CouplingData(state_lin=None, state_quad=StateData(adata.X[5:10]))
        concat_q = CouplingData.concat_collection([q1, q2])
        assert concat_q.state_lin is None
        expected_quad = np.vstack([q1.state_quad.X, q2.state_quad.X])
        np.testing.assert_array_equal(concat_q.state_quad.X, expected_quad)

        # Mix object with linear-only and quadratic-only is invalid
        lin_only = CouplingData(state_lin=StateData(adata.X[:5]), state_quad=None)
        quad_only = CouplingData(state_lin=None, state_quad=StateData(adata.X[5:10]))
        with pytest.raises(ValueError, match="incompatible data"):
            CouplingData.concat_collection([lin_only, quad_only])

    def test_concat_collection_three_objects(self, adata: AnnData) -> None:
        """Concatenate three compatible objects."""
        state_data = StateData(adata.X)
        c1 = CouplingData.init_from_state_data(state_data, n_shared_dims=1)
        c2 = CouplingData.init_from_state_data(state_data, n_shared_dims=1)
        c3 = CouplingData.init_from_state_data(state_data, n_shared_dims=1)

        concatenated = CouplingData.concat_collection([c1, c2, c3])
        total_rows = len(c1) + len(c2) + len(c3)
        assert len(concatenated) == total_rows

        # Check linear and quadratic stacking
        np.testing.assert_array_equal(
            concatenated.state_lin.X, np.vstack([c1.state_lin.X, c2.state_lin.X, c3.state_lin.X])
        )
        np.testing.assert_array_equal(
            concatenated.state_quad.X, np.vstack([c1.state_quad.X, c2.state_quad.X, c3.state_quad.X])
        )

    def test_concat_collection_empty_raises(self) -> None:
        """Concatenating an empty collection should raise an appropriate error."""
        with pytest.raises(ValueError, match="at least one"):
            CouplingData.concat_collection([])

    def test_concat_collection_single_returns_new_instance(self, adata: AnnData) -> None:
        """Concatenating a single object should return a new (but equivalent) CouplingData."""
        c = CouplingData.init_from_state_data(StateData(adata.X), n_shared_dims=2)
        concat_single = CouplingData.concat_collection([c])

        assert concat_single is not c
        assert len(concat_single) == len(c)
        np.testing.assert_array_equal(concat_single.state_lin.X, c.state_lin.X)
        if c.state_quad is not None:
            np.testing.assert_array_equal(concat_single.state_quad.X, c.state_quad.X)
        else:
            assert concat_single.state_quad is None

    def test_concat_collection_preserves_spatial_dimensions(self, adata: AnnData) -> None:
        """Ensure that concatenation does not alter the number of features per component."""
        c1 = CouplingData.init_from_state_data(StateData(adata.X[:10]), n_shared_dims=4)
        c2 = CouplingData.init_from_state_data(StateData(adata.X[10:20]), n_shared_dims=4)
        concat = CouplingData.concat_collection([c1, c2])

        assert concat.state_lin.X.shape[1] == 4
        assert concat.state_quad.X.shape[1] == adata.n_vars - 4

    def test_concat_collection_incompatible_linear_dimensions(self, adata: AnnData) -> None:
        """If linear parts have different number of features, StateData.concat_collection should raise."""
        c1 = CouplingData.init_from_state_data(StateData(adata.X), n_shared_dims=3)  # linear dim = 3
        # Create second object with different linear dimension (2) using custom init
        X_lin = adata.X[:, :2]
        X_quad = adata.X[:, 2:]
        c2 = CouplingData(state_lin=StateData(X_lin), state_quad=StateData(X_quad))
        # Same number of total features, but linear dim differs
        with pytest.raises(ValueError):  # StateData.concat_collection should raise
            CouplingData.concat_collection([c1, c2])
