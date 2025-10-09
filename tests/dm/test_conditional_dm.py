from collections.abc import Sequence

import pytest
from anndata import AnnData

from sc_flow._types import CouplingSpaceReps
from sc_flow.dm._conditional_dm import ConditionalDataManager


class TestConditionalDM:
    @pytest.mark.parametrize("control_key", [None, "is_control", "wrong_key"])
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
    def test_dm_get_condition_data(
        self,
        adata: AnnData,
        control_key: str,
        n_obs: int,
        drug_rep_n_feats: int,
        ko_rep_n_feats: int,
        conditions: dict[str, Sequence[str]],
        conditions_reps: dict[str, str],
        conditions_covariates: dict[str, Sequence[str]] | None,
    ):
        dm = ConditionalDataManager(
            control_key=control_key,
            conditions=conditions,
            conditions_reps=conditions_reps,
            conditions_covariates=conditions_covariates,
        )

        all_cats = tuple(cat for condition in conditions.values() for cat in condition)
        if "wrong_key" in all_cats:
            with pytest.raises(KeyError):
                condition_data = dm.get_condition_data(adata)
            return None

        if "wrong_key" in conditions_reps:
            with pytest.raises(KeyError):
                condition_data = dm.get_condition_data(adata)
            return None

        if conditions_covariates is not None and "wrong_key" in conditions_covariates:
            with pytest.raises(KeyError):
                condition_data = dm.get_condition_data(adata)
            return None

        if conditions_covariates is None:
            all_covariates = ()
        else:
            all_covariates = tuple(cat for condition in conditions_covariates.values() for cat in condition)
        if "wrong_key" in all_covariates:
            with pytest.raises(KeyError):
                condition_data = dm.get_condition_data(adata)
            return None

        condition_data = dm.get_condition_data(adata)

        for cond_realm, cond_reps in condition_data["reps"].items():
            for reps in cond_reps.values():
                if cond_realm == "drug":
                    assert reps.shape == (1, drug_rep_n_feats)
                if cond_realm == "ko":
                    assert reps.shape == (1, ko_rep_n_feats)

        if conditions_covariates is not None:
            for cond_realm, covariates in condition_data["covariates"].items():
                n_combs = len(conditions[cond_realm])
                cond_cat = conditions[cond_realm][0]
                n_feats = len(conditions_covariates[cond_cat])
                assert covariates.shape == (n_obs, n_combs, n_feats)

    @pytest.mark.parametrize("sample_rep", [None, "X_repr"])
    @pytest.mark.parametrize("control_key", [None, "is_control"])
    @pytest.mark.parametrize(
        "coupling_reps",
        [
            None,
            {
                "src_coupling_lin": "X_src_lin",
                "src_coupling_quad": "X_src_quad",
            },
            {
                "src_coupling_lin": "X_src_lin",
                "src_coupling_quad": "X_src_quad",
                "tgt_coupling_lin": "X_tgt_lin",
                "tgt_coupling_quad": "X_tgt_quad_rep",
            },
            {
                "src_coupling_lin": "X_src_lin",
                "src_coupling_quad": "X_src_quad",
                "tgt_coupling_lin": "X_tgt_lin",
                "tgt_coupling_quad": "X_tgt_quad_X",
            },
        ],
    )
    def test_dm_get_coupling_data(
        self,
        adata: AnnData,
        sample_rep: str | None,
        control_key: str,
        coupling_reps: dict[CouplingSpaceReps, str] | None,
    ) -> None:
        dm = ConditionalDataManager(
            sample_rep=sample_rep,
            coupling_reps=coupling_reps,
            control_key=control_key,
        )

        if coupling_reps is None:
            with pytest.raises(ValueError):
                coupling_data = dm.get_coupling_data(adata)
            return None

        if control_key is None:
            with pytest.raises(ValueError):
                coupling_data = dm.get_coupling_data(adata)
            return None

        tgt_quad_rep = coupling_reps.get("tgt_coupling_quad", None)
        if tgt_quad_rep is not None:
            if sample_rep is None and tgt_quad_rep == "X_tgt_quad_rep":
                with pytest.raises(ValueError):
                    coupling_data = dm.get_coupling_data(adata)
                return None

            if sample_rep is not None and tgt_quad_rep == "X_tgt_quad_X":
                with pytest.raises(ValueError):
                    coupling_data = dm.get_coupling_data(adata)
                return None

        coupling_data = dm.get_coupling_data(adata)  # noqa
