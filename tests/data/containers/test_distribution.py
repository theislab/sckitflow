import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from sckitflow.data._mixins import BatchMixin
from sckitflow.data.containers import (
    CategoricalData,
    CouplingData,
    DistributionData,
    MixedTypeData,
    StateData,
)


class DummyEncoder:
    """Dummy encoder that returns a constant (n_samples, 1) array of ones."""

    def transform(self, X):
        n = X.shape[0]
        return np.ones((n, 1))


def _make_categorical_from_obs(obs_df: pd.DataFrame) -> CategoricalData:
    """Create a CategoricalData from a DataFrame, providing dummy encoders for all columns."""
    encoders = {col: DummyEncoder() for col in obs_df.columns}
    return CategoricalData.from_pandas(obs_df, categorical_encoders=encoders)


class TestDistributionData:
    def test_init_minimal(self, adata: AnnData) -> None:
        state = StateData(adata.X)
        dist = DistributionData(state_data=state)

        assert isinstance(dist, DistributionData)
        assert dist.state_data is state
        assert dist.response_data is None
        assert dist.condition_data is None
        assert dist.groups_data is None
        assert dist.source_coupling_data is None
        assert dist.target_coupling_data is None
        assert len(dist) == adata.n_obs

    def test_init_all_components(self, adata: AnnData) -> None:
        state = StateData(adata.X)
        obs_df = adata.obs.copy()

        # Create categorical data with dummy encoders
        cat_data = _make_categorical_from_obs(obs_df)

        response = MixedTypeData(
            categorical_covariates=cat_data,
            continuous_covariates=BatchMixin(adata.obsm),
        )

        condition = MixedTypeData(
            categorical_covariates=cat_data,
            continuous_covariates=None,
        )

        groups = _make_categorical_from_obs(obs_df)

        source_coupling = CouplingData.init_from_state_data(
            StateData(adata.X),
            n_shared_dims=2,
        )

        target_coupling = CouplingData.init_from_state_data(
            StateData(adata.X),
            n_shared_dims=None,
        )

        dist = DistributionData(
            state_data=state,
            response_data=response,
            condition_data=condition,
            groups_data=groups,
            source_coupling_data=source_coupling,
            target_coupling_data=target_coupling,
        )

        assert len(dist) == adata.n_obs

    def test_init_mismatched_n_obs_raises(self, adata: AnnData) -> None:
        state = StateData(adata.X)
        bad_obs = adata.obs.iloc[:-1].copy()
        cat_data = _make_categorical_from_obs(bad_obs)

        response = MixedTypeData(
            categorical_covariates=cat_data,
            continuous_covariates=None,
        )

        with pytest.raises(ValueError):
            DistributionData(
                state_data=state,
                response_data=response,
            )

    @pytest.mark.parametrize("idxs", [slice(0, 5), np.array([0, 2, 4])])
    def test_getitem(self, adata: AnnData, idxs) -> None:
        state = StateData(adata.X)
        obs_df = adata.obs.copy()
        cat_data = _make_categorical_from_obs(obs_df)

        response = MixedTypeData(
            categorical_covariates=cat_data,
            continuous_covariates=BatchMixin(adata.obsm),
        )

        groups = _make_categorical_from_obs(obs_df)

        source_coupling = CouplingData.init_from_state_data(
            StateData(adata.X),
            n_shared_dims=3,
        )

        dist = DistributionData(
            state_data=state,
            response_data=response,
            groups_data=groups,
            source_coupling_data=source_coupling,
        )

        subset = dist[idxs]

        assert isinstance(subset, DistributionData)
        assert len(subset) == len(state[idxs])
        assert subset.response_data is not None
        assert subset.groups_data is not None
        assert subset.source_coupling_data is not None

    def test_ann_df_empty(self, adata: AnnData) -> None:
        state = StateData(adata.X)
        dist = DistributionData(state_data=state)

        ann_df = dist.ann_df

        assert isinstance(ann_df, pd.DataFrame)
        assert ann_df.empty

    def test_ann_df_from_condition_and_groups(self, adata: AnnData) -> None:
        obs_df = adata.obs.copy()

        # Use distinct columns for groups and condition to avoid overwriting
        groups_df = obs_df[["drugA"]].copy()
        groups = _make_categorical_from_obs(groups_df)

        condition_cat_df = obs_df[["drugB"]].copy()
        condition_cat_data = _make_categorical_from_obs(condition_cat_df)
        condition = MixedTypeData(
            categorical_covariates=condition_cat_data,
            continuous_covariates=None,
        )

        dist = DistributionData(
            state_data=StateData(adata.X),
            condition_data=condition,
            groups_data=groups,
        )

        ann_df = dist.ann_df

        # Expected: one column from groups, one from condition
        expected_df = pd.concat([groups.ann_df, condition_cat_data.ann_df], axis=1)
        pd.testing.assert_frame_equal(ann_df, expected_df, check_dtype=False)

    def test_index_matches_ann_df_index(self, adata: AnnData) -> None:
        groups = _make_categorical_from_obs(adata.obs.copy())
        dist = DistributionData(
            state_data=StateData(adata.X),
            groups_data=groups,
        )
        assert dist.index.equals(adata.obs.index)

    def test_repr_contains_components(self, adata: AnnData) -> None:
        groups = _make_categorical_from_obs(adata.obs.copy())
        dist = DistributionData(
            state_data=StateData(adata.X),
            groups_data=groups,
        )

        rep = repr(dist)

        assert "DistributionData" in rep
        assert "state=" in rep
        assert "groups=" in rep
        assert f"n_obs={adata.n_obs}" in rep

    def test_view_on_condition_space_success(self, adata: AnnData) -> None:
        """Successful transformation of a distribution to the condition space."""
        # Create condition_data with both continuous and categorical parts
        continuous_covs = BatchMixin({"X_repr": adata.obsm["X_repr"]})
        # Add a categorical covariate so that after popping the continuous key,
        # the condition_data still exists (categorical remains)
        cat_df = adata.obs[["drugA"]].copy()
        cat_data = _make_categorical_from_obs(cat_df)
        condition_data = MixedTypeData(
            categorical_covariates=cat_data,
            continuous_covariates=continuous_covs,
        )

        # Response and groups remain optional
        obs_df = adata.obs.copy()
        full_cat_data = _make_categorical_from_obs(obs_df)  # for response
        groups_df = obs_df[["drugA"]].copy()
        groups_data = _make_categorical_from_obs(groups_df)

        response_data = MixedTypeData(
            categorical_covariates=full_cat_data,
            continuous_covariates=None,
        )

        dist = DistributionData(
            state_data=StateData(adata.X),
            response_data=response_data,
            condition_data=condition_data,
            groups_data=groups_data,
        )

        new_dist = dist.view_on_condition_space("X_repr")

        # 1. New state_data should be the extracted StateData from condition_data
        assert isinstance(new_dist.state_data, StateData)
        np.testing.assert_array_equal(new_dist.state_data.X, adata.obsm["X_repr"])

        # 2. Condition_data should still exist (because categorical part remains)
        assert new_dist.condition_data is not None
        # The continuous key "X_repr" should be removed
        assert new_dist.condition_data.continuous_covariates is None
        # The categorical part should still be there
        assert new_dist.condition_data.categorical_covariates is cat_data

        # 3. Response_data and groups_data should remain unchanged
        assert new_dist.response_data is response_data
        assert new_dist.groups_data is groups_data

        # 4. Both source and target coupling data should be initialized from new state
        assert new_dist.source_coupling_data is not None
        assert new_dist.target_coupling_data is not None
        assert len(new_dist.source_coupling_data) == len(new_dist.state_data)
        assert len(new_dist.target_coupling_data) == len(new_dist.state_data)

    def test_view_on_condition_space_no_condition(self, adata: AnnData) -> None:
        """TypeError when condition_data is None."""
        dist = DistributionData(state_data=StateData(adata.X))
        with pytest.raises(TypeError, match="Cannot view as condition: no condition data provided."):
            dist.view_on_condition_space("any_key")

    def test_view_on_condition_space_key_not_found(self, adata: AnnData) -> None:
        """KeyError when state_key does not exist in condition_data."""
        continuous_covs = BatchMixin({"X_repr": adata.obsm["X_repr"]})
        condition_data = MixedTypeData(continuous_covariates=continuous_covs)
        dist = DistributionData(
            state_data=StateData(adata.X),
            condition_data=condition_data,
        )

        with pytest.raises(KeyError, match="Key missing_key not found"):
            dist.view_on_condition_space("missing_key")

    def test_view_on_condition_space_pops_key_and_preserves_other_keys(self, adata: AnnData) -> None:
        """Ensure only the specified key is popped, other keys remain."""
        # Add a second key if it doesn't exist (use a random array)
        obsm_dict = dict(adata.obsm)
        if "X_umap" not in obsm_dict:
            obsm_dict["X_umap"] = np.random.randn(adata.n_obs, 2)
        continuous_covs = BatchMixin(
            {
                "X_repr": obsm_dict["X_repr"],
                "X_umap": obsm_dict["X_umap"],
            }
        )
        condition_data = MixedTypeData(continuous_covariates=continuous_covs)
        dist = DistributionData(state_data=StateData(adata.X), condition_data=condition_data)

        new_dist = dist.view_on_condition_space("X_repr")

        # "X_repr" removed
        assert "X_repr" not in new_dist.condition_data.continuous_covariates.mapping
        # "X_umap" still present
        assert "X_umap" in new_dist.condition_data.continuous_covariates.mapping
        np.testing.assert_array_equal(
            new_dist.condition_data.continuous_covariates.mapping["X_umap"],
            continuous_covs.mapping["X_umap"],
        )

    def test_view_on_condition_space_coupling_initialization(self, adata: AnnData) -> None:
        """Check that coupling data are correctly initialized from extracted state."""
        # Add a random feature if needed
        feature = np.random.randn(adata.n_obs, 5)
        continuous_covs = BatchMixin({"feature": feature})
        condition_data = MixedTypeData(continuous_covariates=continuous_covs)
        dist = DistributionData(state_data=StateData(adata.X), condition_data=condition_data)

        new_dist = dist.view_on_condition_space("feature")

        assert len(new_dist.source_coupling_data) == adata.n_obs
        assert len(new_dist.target_coupling_data) == adata.n_obs
        assert new_dist.source_coupling_data is not new_dist.target_coupling_data

    def test_concat_collection_minimal(self, adata: AnnData) -> None:
        """Concatenate two DistributionData objects with only state_data."""
        s1 = StateData(adata.X[:10])
        s2 = StateData(adata.X[10:20])
        d1 = DistributionData(state_data=s1)
        d2 = DistributionData(state_data=s2)

        concatenated = DistributionData.concat_collection([d1, d2])

        assert len(concatenated) == 20
        np.testing.assert_array_equal(concatenated.state_data.X, np.vstack([s1.X, s2.X]))
        assert concatenated.response_data is None
        assert concatenated.condition_data is None
        assert concatenated.groups_data is None
        assert concatenated.source_coupling_data is None
        assert concatenated.target_coupling_data is None

    def test_concat_collection_full(self, adata: AnnData) -> None:
        """Concatenate two DistributionData objects with all components present."""

        # Build two compatible distributions
        def make_dist(start, end):
            X_chunk = adata.X[start:end]
            obs_chunk = adata.obs.iloc[start:end]
            obsm_chunk = {k: v[start:end] for k, v in adata.obsm.items()}

            state = StateData(X_chunk)

            cat_data = _make_categorical_from_obs(obs_chunk)
            response = MixedTypeData(
                categorical_covariates=cat_data,
                continuous_covariates=BatchMixin(obsm_chunk),
            )
            condition = MixedTypeData(
                categorical_covariates=cat_data,
                continuous_covariates=None,
            )
            groups = _make_categorical_from_obs(obs_chunk)

            source_coupling = CouplingData.init_from_state_data(StateData(X_chunk), n_shared_dims=2)
            target_coupling = CouplingData.init_from_state_data(StateData(X_chunk), n_shared_dims=None)

            return DistributionData(
                state_data=state,
                response_data=response,
                condition_data=condition,
                groups_data=groups,
                source_coupling_data=source_coupling,
                target_coupling_data=target_coupling,
            )

        d1 = make_dist(0, 10)
        d2 = make_dist(10, 20)

        concatenated = DistributionData.concat_collection([d1, d2])

        assert len(concatenated) == 20

        # Check state_data
        np.testing.assert_array_equal(concatenated.state_data.X, np.vstack([d1.state_data.X, d2.state_data.X]))

        # Check response_data (MixedTypeData)
        assert concatenated.response_data is not None
        assert len(concatenated.response_data) == 20

        # Check condition_data
        assert concatenated.condition_data is not None
        assert len(concatenated.condition_data) == 20

        # Check groups_data
        assert concatenated.groups_data is not None
        assert len(concatenated.groups_data) == 20

        # Check source_coupling_data
        assert concatenated.source_coupling_data is not None
        np.testing.assert_array_equal(
            concatenated.source_coupling_data.state_lin.X,
            np.vstack([d1.source_coupling_data.state_lin.X, d2.source_coupling_data.state_lin.X]),
        )

        # Check target_coupling_data
        assert concatenated.target_coupling_data is not None
        np.testing.assert_array_equal(
            concatenated.target_coupling_data.state_lin.X,
            np.vstack([d1.target_coupling_data.state_lin.X, d2.target_coupling_data.state_lin.X]),
        )

    def test_concat_collection_three_objects(self, adata: AnnData) -> None:
        """Concatenate three minimal distributions."""
        states = [StateData(adata.X[i * 5 : (i + 1) * 5]) for i in range(3)]
        dists = [DistributionData(state_data=s) for s in states]

        concatenated = DistributionData.concat_collection(dists)

        assert len(concatenated) == 15
        expected_X = np.vstack([s.X for s in states])
        np.testing.assert_array_equal(concatenated.state_data.X, expected_X)

    def test_concat_collection_incompatible_response(self, adata: AnnData) -> None:
        """Mismatch in presence of response_data raises ValueError."""
        d1 = DistributionData(state_data=StateData(adata.X[:10]))
        cat_data = _make_categorical_from_obs(adata.obs.iloc[10:20])
        response = MixedTypeData(categorical_covariates=cat_data)
        d2 = DistributionData(state_data=StateData(adata.X[10:20]), response_data=response)

        with pytest.raises(ValueError, match="incompatible objects"):
            DistributionData.concat_collection([d1, d2])

    def test_concat_collection_incompatible_condition(self, adata: AnnData) -> None:
        """Mismatch in presence of condition_data raises ValueError."""
        d1 = DistributionData(state_data=StateData(adata.X[:10]))
        cat_data = _make_categorical_from_obs(adata.obs.iloc[10:20])
        condition = MixedTypeData(categorical_covariates=cat_data)
        d2 = DistributionData(state_data=StateData(adata.X[10:20]), condition_data=condition)

        with pytest.raises(ValueError, match="incompatible objects"):
            DistributionData.concat_collection([d1, d2])

    def test_concat_collection_incompatible_groups(self, adata: AnnData) -> None:
        """Mismatch in presence of groups_data raises ValueError."""
        d1 = DistributionData(state_data=StateData(adata.X[:10]))
        groups = _make_categorical_from_obs(adata.obs.iloc[10:20])
        d2 = DistributionData(state_data=StateData(adata.X[10:20]), groups_data=groups)

        with pytest.raises(ValueError, match="incompatible objects"):
            DistributionData.concat_collection([d1, d2])

    def test_concat_collection_incompatible_source_coupling(self, adata: AnnData) -> None:
        """Mismatch in presence of source_coupling_data raises ValueError."""
        d1 = DistributionData(state_data=StateData(adata.X[:10]))
        source_coupling = CouplingData.init_from_state_data(StateData(adata.X[10:20]))
        d2 = DistributionData(state_data=StateData(adata.X[10:20]), source_coupling_data=source_coupling)

        with pytest.raises(ValueError, match="incompatible objects"):
            DistributionData.concat_collection([d1, d2])

    def test_concat_collection_incompatible_target_coupling(self, adata: AnnData) -> None:
        """Mismatch in presence of target_coupling_data raises ValueError."""
        d1 = DistributionData(state_data=StateData(adata.X[:10]))
        target_coupling = CouplingData.init_from_state_data(StateData(adata.X[10:20]))
        d2 = DistributionData(state_data=StateData(adata.X[10:20]), target_coupling_data=target_coupling)

        with pytest.raises(ValueError, match="incompatible objects"):
            DistributionData.concat_collection([d1, d2])

    def test_concat_collection_empty_raises(self) -> None:
        """Empty collection raises ValueError."""
        with pytest.raises(ValueError, match="at least one"):
            DistributionData.concat_collection([])

    def test_concat_collection_single_returns_new_instance(self, adata: AnnData) -> None:
        """Single-element concatenation returns a new instance (not the same reference)."""
        original = DistributionData(state_data=StateData(adata.X))
        result = DistributionData.concat_collection([original])

        assert result is not original
        np.testing.assert_array_equal(result.state_data.X, original.state_data.X)
        assert result.response_data is original.response_data  # None, same reference OK
        assert result.condition_data is original.condition_data
        assert result.groups_data is original.groups_data
        assert result.source_coupling_data is original.source_coupling_data
        assert result.target_coupling_data is original.target_coupling_data


class TestDistributionDataNoneState:
    """No target state available: prediction-without-target-states use case."""

    def test_len_falls_back_to_condition_data(self, adata: AnnData) -> None:
        obs_df = adata.obs.copy()
        cat_data = _make_categorical_from_obs(obs_df)
        condition = MixedTypeData(categorical_covariates=cat_data, continuous_covariates=None)

        dist = DistributionData(state_data=None, condition_data=condition)

        assert dist.state_data is None
        assert len(dist) == adata.n_obs

    def test_len_falls_back_to_groups_data(self, adata: AnnData) -> None:
        groups = _make_categorical_from_obs(adata.obs.copy())
        dist = DistributionData(state_data=None, groups_data=groups)

        assert len(dist) == adata.n_obs

    def test_len_raises_when_everything_is_none(self) -> None:
        dist = DistributionData(state_data=None)
        with pytest.raises(ValueError, match="Cannot determine length"):
            len(dist)

    def test_post_init_does_not_crash_without_state(self, adata: AnnData) -> None:
        groups = _make_categorical_from_obs(adata.obs.copy())
        # should not raise, even though groups_data is set and state_data is None
        dist = DistributionData(state_data=None, groups_data=groups)
        assert dist.state_data is None

    def test_getitem_keeps_state_none(self, adata: AnnData) -> None:
        groups = _make_categorical_from_obs(adata.obs.copy())
        dist = DistributionData(state_data=None, groups_data=groups)

        subset = dist[slice(0, 5)]

        assert subset.state_data is None
        assert len(subset) == 5

    def test_concat_collection_all_none_state(self, adata: AnnData) -> None:
        groups1 = _make_categorical_from_obs(adata.obs.iloc[:10])
        groups2 = _make_categorical_from_obs(adata.obs.iloc[10:20])
        d1 = DistributionData(state_data=None, groups_data=groups1)
        d2 = DistributionData(state_data=None, groups_data=groups2)

        concatenated = DistributionData.concat_collection([d1, d2])

        assert concatenated.state_data is None
        assert len(concatenated) == 20

    def test_concat_collection_incompatible_state(self, adata: AnnData) -> None:
        """Mismatch in presence of state_data raises ValueError, same as other optional fields."""
        d1 = DistributionData(state_data=None)
        d2 = DistributionData(state_data=StateData(adata.X[:10]))

        with pytest.raises(ValueError, match="incompatible objects"):
            DistributionData.concat_collection([d1, d2])
