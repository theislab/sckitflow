import numpy as np
import pytest

from sc_flow.data.containers import StateData


class TestStateData:
    def test_init_and_len(self) -> None:
        X = np.random.randn(10, 3)

        state = StateData(X)

        assert isinstance(state, StateData)
        assert state.X is X
        assert len(state) == 10

    def test_len_one_dimensional(self) -> None:
        X = np.random.randn(5)

        state = StateData(X)

        assert len(state) == 5

    @pytest.mark.parametrize("idxs", [slice(0, 5), np.array([0, 2, 4])])
    def test_getitem(self, idxs) -> None:
        X = np.random.randn(10, 3)

        state = StateData(X)
        subset = state[idxs]

        assert isinstance(subset, StateData)
        np.testing.assert_array_equal(subset.X, X[idxs])
        assert len(subset) == X[idxs].shape[0]

    def test_getitem_preserves_spatial_dims(self) -> None:
        X = np.random.randn(10, 4, 2)

        state = StateData(X)
        subset = state[:3]

        assert subset.X.shape == (3, 4, 2)

    def test_repr_contains_shape_information(self) -> None:
        X = np.random.randn(7, 5)

        state = StateData(X)
        rep = repr(state)

        assert "StateData" in rep
        assert "n_obs=7" in rep
        assert "spatial_dims=(5,)" in rep

    def test_repr_no_spatial_dims_for_vector(self) -> None:
        X = np.random.randn(6)

        state = StateData(X)
        rep = repr(state)

        assert "StateData" in rep
        assert "n_obs=6" in rep
        assert "spatial_dims=()" in rep
