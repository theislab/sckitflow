import numpy as np
import pytest
from anndata import AnnData

from sc_flow.data._mixins import BatchMixin
from sc_flow.data.containers import CategoricalData, MixedTypeData, StateData


class DummyEncoder:
    """Dummy encoder that returns a constant (n_samples, 1) array of ones."""

    def transform(self, X):
        n = X.shape[0]
        return np.ones((n, 1))


def _make_categorical_from_obs(obs_df):
    """Create a CategoricalData from a DataFrame, providing dummy encoders for all columns."""
    encoders = {col: DummyEncoder() for col in obs_df.columns}
    return CategoricalData.from_pandas(obs_df, categorical_encoders=encoders)


class TestMixedTypeData:
    @pytest.mark.parametrize("use_categorical_covariates", [True, False])
    @pytest.mark.parametrize("use_continuous_covariates", [True, False])
    def test_init(
        self,
        adata: AnnData,
        use_categorical_covariates: bool,
        use_continuous_covariates: bool,
    ) -> None:
        if not use_categorical_covariates and not use_continuous_covariates:
            with pytest.raises(ValueError, match="must contain at least one covariate container"):
                MixedTypeData()
            return

        categorical_data = _make_categorical_from_obs(adata.obs) if use_categorical_covariates else None
        continuous_data = BatchMixin(adata.obsm) if use_continuous_covariates else None

        mixed_data = MixedTypeData(
            categorical_covariates=categorical_data,
            continuous_covariates=continuous_data,
        )

        assert mixed_data.categorical_covariates is categorical_data
        assert mixed_data.continuous_covariates is continuous_data
        assert len(mixed_data) == adata.n_obs

    def test_init_shape_mismatch(self, adata: AnnData) -> None:
        categorical_data = _make_categorical_from_obs(adata.obs.iloc[:-1])
        continuous_data = BatchMixin(adata.obsm)

        with pytest.raises(ValueError, match="Shape mismatch"):
            MixedTypeData(
                categorical_covariates=categorical_data,
                continuous_covariates=continuous_data,
            )

    def test_getitem_slice(self, adata: AnnData) -> None:
        categorical_data = _make_categorical_from_obs(adata.obs)
        continuous_data = BatchMixin(adata.obsm)

        mixed_data = MixedTypeData(
            categorical_covariates=categorical_data,
            continuous_covariates=continuous_data,
        )

        subset = mixed_data[:5]

        assert isinstance(subset, MixedTypeData)
        assert len(subset) == 5
        assert subset.categorical_covariates is not None
        assert subset.continuous_covariates is not None

    def test_getitem_array(self, adata: AnnData) -> None:
        categorical_data = _make_categorical_from_obs(adata.obs)
        continuous_data = BatchMixin(adata.obsm)

        mixed_data = MixedTypeData(
            categorical_covariates=categorical_data,
            continuous_covariates=continuous_data,
        )

        idxs = np.array([0, 2, 4])
        subset = mixed_data[idxs]

        assert len(subset) == len(idxs)

    def test_view_as_state_data_success(self, adata: AnnData) -> None:
        """view_as_state_data returns StateData when key exists in continuous covariates."""
        # Use an existing key, e.g., "X_repr"
        continuous_data = BatchMixin({"X_repr": adata.obsm["X_repr"]})
        mixed = MixedTypeData(continuous_covariates=continuous_data)

        state = mixed.view_as_state_data("X_repr")
        assert isinstance(state, StateData)
        np.testing.assert_array_equal(state.X, adata.obsm["X_repr"])

    def test_view_as_state_data_key_error(self, adata: AnnData) -> None:
        """KeyError if key not found in continuous nor categorical."""
        continuous_data = BatchMixin({"X_repr": adata.obsm["X_repr"]})
        mixed = MixedTypeData(continuous_covariates=continuous_data)

        with pytest.raises(KeyError, match="Key missing_key not found"):
            mixed.view_as_state_data("missing_key")

    def test_view_as_state_data_categorical_not_implemented(self, adata: AnnData) -> None:
        """NotImplementedError when key is in categorical covariates."""
        # Use a real column from adata.obs, e.g., "drugA"
        cat_data = _make_categorical_from_obs(adata.obs[["drugA"]])
        mixed = MixedTypeData(categorical_covariates=cat_data)

        with pytest.raises(NotImplementedError, match="State conversion is not implemented for categorical covariates"):
            mixed.view_as_state_data("drugA")

    # Removed test_view_as_state_data_ndim_error because BatchMixin now rejects 1D arrays

    def test_absorb_state_data_no_continuous(self, adata: AnnData) -> None:
        """TypeError if continuous_covariates is None."""
        cat_data = _make_categorical_from_obs(adata.obs)
        mixed = MixedTypeData(categorical_covariates=cat_data)

        state = StateData(np.random.randn(adata.n_obs, 5))
        with pytest.raises(TypeError, match="Cannot absorb state data: no continuous covariates present"):
            mixed.absorb_state_data("new_key", state)

    def test_absorb_state_data_key_exists_no_override(self, adata: AnnData) -> None:
        """ValueError when key already exists and allow_override=False."""
        continuous_data = BatchMixin({"existing": adata.obsm["X_repr"]})
        mixed = MixedTypeData(continuous_covariates=continuous_data)
        state = StateData(np.random.randn(adata.n_obs, 10))

        with pytest.raises(ValueError, match="Key existing already present in the data"):
            mixed.absorb_state_data("existing", state, allow_override=False)

    def test_absorb_state_data_length_mismatch(self, adata: AnnData) -> None:
        """ValueError when state_data length != len(self)."""
        continuous_data = BatchMixin({"existing": adata.obsm["X_repr"]})
        mixed = MixedTypeData(continuous_covariates=continuous_data)
        state = StateData(np.random.randn(adata.n_obs - 5, 10))  # wrong length

        with pytest.raises(ValueError, match="Number of observations in state_data .* does not match"):
            mixed.absorb_state_data("new_key", state)

    def test_absorb_state_data_success(self, adata: AnnData) -> None:
        """Successful absorption of state data."""
        continuous_data = BatchMixin({"original": adata.obsm["X_repr"]})
        mixed = MixedTypeData(continuous_covariates=continuous_data)
        new_state = StateData(np.random.randn(adata.n_obs, 8))

        new_mixed = mixed.absorb_state_data("new_key", new_state, allow_override=False)

        assert "new_key" not in mixed.continuous_covariates.mapping
        assert "new_key" in new_mixed.continuous_covariates.mapping
        np.testing.assert_array_equal(new_mixed.continuous_covariates.mapping["new_key"], new_state.X)
        np.testing.assert_array_equal(new_mixed.continuous_covariates.mapping["original"], adata.obsm["X_repr"])
        assert new_mixed.categorical_covariates is mixed.categorical_covariates

    def test_absorb_state_data_allow_override(self, adata: AnnData) -> None:
        """Override existing key when allow_override=True."""
        continuous_data = BatchMixin({"original": adata.obsm["X_repr"]})
        mixed = MixedTypeData(continuous_covariates=continuous_data)
        new_state = StateData(np.random.randn(adata.n_obs, 8))

        new_mixed = mixed.absorb_state_data("original", new_state, allow_override=True)

        np.testing.assert_array_equal(new_mixed.continuous_covariates.mapping["original"], new_state.X)

    def test_pop_key_no_continuous(self, adata: AnnData) -> None:
        """TypeError when continuous_covariates is None."""
        cat_data = _make_categorical_from_obs(adata.obs)
        mixed = MixedTypeData(categorical_covariates=cat_data)

        with pytest.raises(TypeError, match="Cannot remove key: no continuous covariates present"):
            mixed.pop_key("any_key")

    def test_pop_key_categorical_not_implemented(self, adata: AnnData) -> None:
        """NotImplementedError when key belongs to categorical covariates."""
        cat_data = _make_categorical_from_obs(adata.obs[["drugA"]])
        continuous_data = BatchMixin({"some_key": adata.obsm["X_repr"]})
        mixed = MixedTypeData(categorical_covariates=cat_data, continuous_covariates=continuous_data)

        with pytest.raises(NotImplementedError, match="State conversion is not implemented for categorical covariates"):
            mixed.pop_key("drugA")

    def test_pop_key_missing_key(self, adata: AnnData) -> None:
        """KeyError when key not found in continuous covariates."""
        continuous_data = BatchMixin({"existing": adata.obsm["X_repr"]})
        mixed = MixedTypeData(continuous_covariates=continuous_data)

        with pytest.raises(KeyError, match="Key missing_key not found"):
            mixed.pop_key("missing_key")

    def test_pop_key_success_keep_categorical(self, adata: AnnData) -> None:
        """Remove a key and keep categorical if present."""
        cat_data = _make_categorical_from_obs(adata.obs)
        continuous_data = BatchMixin({"a": adata.obsm["X_repr"], "b": adata.obsm["X_repr"]})
        mixed = MixedTypeData(categorical_covariates=cat_data, continuous_covariates=continuous_data)

        new_mixed = mixed.pop_key("a")

        assert "a" not in new_mixed.continuous_covariates.mapping
        assert "b" in new_mixed.continuous_covariates.mapping
        assert new_mixed.categorical_covariates is cat_data
        assert len(new_mixed) == len(mixed)

    def test_pop_key_returns_none_when_empty(self, adata: AnnData) -> None:
        """Return None when both continuous and categorical become empty."""
        continuous_data = BatchMixin({"only": adata.obsm["X_repr"]})
        mixed = MixedTypeData(continuous_covariates=continuous_data)  # categorical is None

        result = mixed.pop_key("only")

        assert result is None

    def test_pop_key_keeps_continuous_when_categorical_present(self, adata: AnnData) -> None:
        """When continuous becomes empty but categorical still present, return new MixedTypeData."""
        cat_data = _make_categorical_from_obs(adata.obs)
        continuous_data = BatchMixin({"only": adata.obsm["X_repr"]})
        mixed = MixedTypeData(categorical_covariates=cat_data, continuous_covariates=continuous_data)

        new_mixed = mixed.pop_key("only")

        assert new_mixed is not None
        assert new_mixed.continuous_covariates is None  # continuous becomes None
        assert new_mixed.categorical_covariates is cat_data
