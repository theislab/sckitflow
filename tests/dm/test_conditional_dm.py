from collections.abc import Sequence

import pytest
from anndata import AnnData

from sc_flow.dm._conditional_dm import ConditionalDataManager


class TestConditionalDM:
    @pytest.mark.parametrize("used_control_key", [None, "is_control", "wrong_key"])
    @pytest.mark.parametrize(
        "conditions",
        [
            {"drug": ("drugA", "drugB"), "ko": ("koA", "koB")},
            {"drug": ("drugA", "drugB", "wrong_key"), "ko": ("koA", "koB")},
        ],
    )
    @pytest.mark.parametrize(
        "conditions_reps",
        [
            {"drug": "drug", "ko": "ko"},
            {"drug": "drug", "ko": "ko", "wrong_key": "wrong_key"},
        ],
    )
    @pytest.mark.parametrize(
        "conditions_covariates",
        [
            None,
            {
                "drugA": ("drugA_time", "drugA_dose"),
                "drugB": ("drugB_time", "drugB_dose"),
                "koA": ("koA_time", "koA_dose"),
                "koB": ("koB_time", "koB_dose"),
            },
            {
                "wrong_key": ("drugA_time", "drugA_dose", "wrong_key"),
                "drugB": ("drugB_time", "drugB_dose"),
                "koA": ("koA_time", "koA_dose"),
                "koB": ("koB_time", "koB_dose"),
            },
            {
                "drugA": ("drugA_time", "drugA_dose", "wrong_key"),
                "drugB": ("drugB_time", "drugB_dose"),
                "koA": ("koA_time", "koA_dose"),
                "koB": ("koB_time", "koB_dose"),
            },
        ],
    )
    @pytest.mark.parametrize("groups_obs_keys", [None, ("source_split",)])
    def test_dm_get_dataset(
        self,
        adata: AnnData,
        used_control_key: str,
        n_obs: int,
        drug_rep_n_feats: int,
        ko_rep_n_feats: int,
        conditions: dict[str, Sequence[str]],
        conditions_reps: dict[str, str],
        conditions_covariates: dict[str, Sequence[str]] | None,
        groups_obs_keys: Sequence[str] | None,
    ):
        dm = ConditionalDataManager(
            control_key=used_control_key,
            conditions=conditions,
            conditions_reps=conditions_reps,
            conditions_covariates=conditions_covariates,
            groups_obs_keys=groups_obs_keys,
        )

        if used_control_key == "wrong_key":
            with pytest.raises(KeyError):
                dataset = dm.get_dataset(adata)
            return None

        all_cats = tuple(cat for condition in conditions.values() for cat in condition)
        if "wrong_key" in all_cats:
            with pytest.raises(KeyError):
                dataset = dm.get_dataset(adata)
            return None

        if "wrong_key" in conditions_reps:
            with pytest.raises(ValueError):
                dataset = dm.get_dataset(adata)
            return None

        if conditions_covariates is not None and "wrong_key" in conditions_covariates:
            with pytest.raises(KeyError):
                dataset = dm.get_dataset(adata)
            return None

        if conditions_covariates is None:
            all_covariates = ()
        else:
            all_covariates = tuple(cat for condition in conditions_covariates.values() for cat in condition)
        if "wrong_key" in all_covariates:
            with pytest.raises(KeyError):
                dataset = dm.get_dataset(adata)
            return None

        dataset = dm.get_dataset(adata)

        # retrieving and verifying condition data
        condition_data = dataset.condition_data
        for cond_realm, cond_reps in condition_data.condition_reps.items():
            for reps in cond_reps.values():
                if cond_realm == "drug":
                    assert reps.shape == (1, drug_rep_n_feats)
                if cond_realm == "ko":
                    assert reps.shape == (1, ko_rep_n_feats)
        if conditions_covariates is not None:
            for cond_realm, covariates in condition_data.condition_covariates.items():
                n_combs = len(conditions[cond_realm])
                cond_cat = conditions[cond_realm][0]
                n_feats = len(conditions_covariates[cond_cat])
                assert covariates.shape == (n_obs, n_combs, n_feats)

        # retrieving and verifying indices
        populations_indices = dataset.group_to_populations_indices
        # when no population indices specified, we have only one group
        if groups_obs_keys is None:
            assert len(populations_indices) == 1
            assert None in populations_indices
