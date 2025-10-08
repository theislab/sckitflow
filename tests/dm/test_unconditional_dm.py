from collections.abc import Sequence

import pytest
from anndata import AnnData

from sc_flow.dm._unconditional_dm import UnconditionalDataManager


class TestUnconditionalDM:
    @pytest.mark.parametrize("sample_rep", [None, "X_repr", "wrong_key"])
    def test_dm_get_state_data(
        self,
        adata: AnnData,
        n_obs: int,
        n_genes: int,
        n_feats_obsm_repr: int,
        sample_rep: str | None,
    ) -> None:
        dm = UnconditionalDataManager(sample_rep=sample_rep)

        if sample_rep == "wrong_key":
            with pytest.raises(KeyError):
                state_data = dm.get_state_data(adata)
            return None
        state_data = dm.get_state_data(adata)
        if sample_rep == "X_repr":
            assert state_data.shape == (n_obs, n_feats_obsm_repr)
        else:
            assert state_data.shape == (n_obs, n_genes)

    @pytest.mark.parametrize(
        "categorical_target_covariates",
        [
            None,
            {"target": "label"},
            {"target": "one-hot"},
            {"target": "wrong_encoding"},
            {"wrong_key": "label"},
            {"wrong_key": "one-hot"},
            {"wrong_key": "wrong_encoding"},
        ],
    )
    @pytest.mark.parametrize("continuous_target_covariates", [None, ("target_variable",), ("wrong_key",)])
    def test_dm_get_target_data(
        self,
        adata: AnnData,
        n_obs: int,
        nunique_targets: int,
        continuous_target_dim: int,
        categorical_target_covariates: dict[str, str],
        continuous_target_covariates: Sequence[str],
    ) -> None:
        dm = UnconditionalDataManager(
            categorical_target_covariates=categorical_target_covariates,
            continuous_target_covariates=continuous_target_covariates,
        )

        if categorical_target_covariates is not None and "wrong_key" in categorical_target_covariates:
            with pytest.raises(KeyError):
                target_data = dm.get_target_data(adata)
            return None

        if categorical_target_covariates is not None and "wrong_encoding" in categorical_target_covariates.values():
            with pytest.raises(ValueError):
                target_data = dm.get_target_data(adata)
            return None

        if continuous_target_covariates is not None and "wrong_key" in continuous_target_covariates:
            with pytest.raises(KeyError):
                target_data = dm.get_target_data(adata)
            return None
        target_data = dm.get_target_data(adata)

        if categorical_target_covariates is not None:
            assert "target" in target_data
            if categorical_target_covariates["target"] == "label":
                assert target_data["target"].shape == (n_obs,)
            else:
                assert target_data["target"].shape == (n_obs, nunique_targets)

        if continuous_target_covariates is not None:
            assert "target_variable" in target_data
            assert target_data["target_variable"].shape == (n_obs, continuous_target_dim)
