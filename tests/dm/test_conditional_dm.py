from collections.abc import Sequence

import pytest
from anndata import AnnData

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
            {"drugA": ("drugA_time", "drugA_dose"), "drugB": ("drugB_time", "drugB_dose")},
            {"wrong_key": ("drugA_time", "drugA_dose", "wrong_key"), "drugB": ("drugB_time", "drugB_dose")},
            {"drugA": ("drugA_time", "drugA_dose", "wrong_key"), "drugB": ("drugB_time", "drugB_dose")},
        ],
    )
    @pytest.mark.parametrie("split_covariates", [None, "source_split", "wrong_key"])
    def test_dm_get_condition_data(
        self,
        adata: AnnData,
        control_key: str,
        conditions: dict[str, Sequence[str]],
        conditions_reps: dict[str, str],
        conditions_covariates: dict[str, Sequence[str]],
    ):
        dm = ConditionalDataManager(
            control_key=control_key,
            conditions=conditions,
            conditions_reps=conditions_reps,
            conditions_covariates=conditions_covariates,
        )

        all_cats = tuple(cat for condition in conditions.values() for cat in condition)
        print(all_cats)
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

        print(condition_data.keys())
