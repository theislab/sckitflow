import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from sc_flow.data._mixins import BatchMixin
from sc_flow.data.containers import (
    CategoricalData,
    CouplingData,
    DistributionData,
    MixedTypeData,
    StateData,
)


class TestDistributionData:
    def test_init_minimal(self, adata: AnnData) -> None:
        state = StateData(adata.X)

        dist = DistributionData(state_data=state)

        assert isinstance(dist, DistributionData)
        assert dist.state_data is state
        assert dist.target_data is None
        assert dist.condition_data is None
        assert dist.groups_data is None
        assert dist.source_coupling_data is None
        assert dist.target_coupling_data is None
        assert len(dist) == adata.n_obs

    def test_init_all_components(self, adata: AnnData) -> None:
        state = StateData(adata.X)

        obs_df = adata.obs.copy()

        target = MixedTypeData(
            categorical_covariates=CategoricalData(obs_df),
            continuous_covariates=BatchMixin(adata.obsm),
        )

        condition = MixedTypeData(
            categorical_covariates=CategoricalData(obs_df),
            continuous_covariates=None,
        )

        groups = CategoricalData(obs_df)

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
            target_data=target,
            condition_data=condition,
            groups_data=groups,
            source_coupling_data=source_coupling,
            target_coupling_data=target_coupling,
        )

        assert len(dist) == adata.n_obs

    def test_init_mismatched_n_obs_raises(self, adata: AnnData) -> None:
        state = StateData(adata.X)
        bad_obs = adata.obs.iloc[:-1]

        target = MixedTypeData(
            categorical_covariates=CategoricalData(bad_obs),
            continuous_covariates=None,
        )

        with pytest.raises(ValueError):
            DistributionData(
                state_data=state,
                target_data=target,
            )

    @pytest.mark.parametrize("idxs", [slice(0, 5), np.array([0, 2, 4])])
    def test_getitem(self, adata: AnnData, idxs) -> None:
        state = StateData(adata.X)

        obs_df = adata.obs.copy()

        target = MixedTypeData(
            categorical_covariates=CategoricalData(obs_df),
            continuous_covariates=BatchMixin(adata.obsm),
        )

        groups = CategoricalData(obs_df)

        source_coupling = CouplingData.init_from_state_data(
            StateData(adata.X),
            n_shared_dims=3,
        )

        dist = DistributionData(
            state_data=state,
            target_data=target,
            groups_data=groups,
            source_coupling_data=source_coupling,
        )

        subset = dist[idxs]

        assert isinstance(subset, DistributionData)
        assert len(subset) == len(state[idxs])
        assert subset.target_data is not None
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

        condition = MixedTypeData(
            categorical_covariates=CategoricalData(obs_df),
            continuous_covariates=None,
        )

        groups = CategoricalData(obs_df)

        dist = DistributionData(
            state_data=StateData(adata.X),
            condition_data=condition,
            groups_data=groups,
        )

        ann_df = dist.ann_df

        pd.testing.assert_frame_equal(
            ann_df,
            pd.concat([obs_df, obs_df], axis=1),
        )

    def test_index_matches_ann_df_index(self, adata: AnnData) -> None:
        groups = CategoricalData(adata.obs.copy())

        dist = DistributionData(
            state_data=StateData(adata.X),
            groups_data=groups,
        )

        assert dist.index.equals(adata.obs.index)

    def test_repr_contains_components(self, adata: AnnData) -> None:
        dist = DistributionData(
            state_data=StateData(adata.X),
            groups_data=CategoricalData(adata.obs.copy()),
        )

        rep = repr(dist)

        assert "DistributionData" in rep
        assert "state=" in rep
        assert "groups=" in rep
        assert f"n_obs={adata.n_obs}" in rep
