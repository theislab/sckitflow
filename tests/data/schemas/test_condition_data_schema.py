from collections.abc import Collection

import pytest
from anndata.acc import A

from sckitflow.data.schemas import ConditionDataSchema

from ..shared import verify_categorical_data, verify_mixin  # noqa

inval_key: str = "invalid_key"


class TestConditionDataSchema:
    @staticmethod
    def _is_valid_args(
        conditions: dict[str, Collection[str]],
        categorical_reps: dict[str, str],
    ) -> bool:
        if conditions is None and categorical_reps is None:
            return True
        elif conditions is not None and categorical_reps is not None:
            return set(conditions.keys()) == set(categorical_reps.keys())
        else:
            return False

    @pytest.mark.parametrize(
        "conditions",
        [
            None,
            {
                "drug": ["drugA", "drugB"],
            },
            {
                "drug": ["drugA", "drugB"],
                "ko": ["koA", "koB"],
            },
        ],
    )
    @pytest.mark.parametrize(
        "categorical_reps",
        [
            None,
            {"drug": "drug"},
            {"drug": "drug", "ko": "ko"},
            {inval_key: "drug", "ko": "ko"},
        ],
    )
    @pytest.mark.parametrize(
        "continuous_conditions",
        [
            None,
            [A.obsm["drugA_time"], A.obsm["drugA_dose"]],
        ],
    )
    def test_init(
        self,
        conditions: dict[str, Collection[str]],
        categorical_reps: dict[str, str],
        continuous_conditions: Collection[str] | None,
    ) -> None:
        # failure at initialization
        is_valid_args = self._is_valid_args(conditions, categorical_reps)
        if (
            (conditions is not None and categorical_reps is None)
            or (conditions is None and categorical_reps is not None)
            or not is_valid_args
        ):
            with pytest.raises(ValueError):
                # with pytest.raises(ValueError, match="The following reference columns are missing"):
                schema = ConditionDataSchema(
                    categorical_conditions=conditions,
                    categorical_reps=categorical_reps,
                    continuous_conditions=continuous_conditions,
                )
            return None
        # init schema
        schema = ConditionDataSchema(
            categorical_conditions=conditions,
            categorical_reps=categorical_reps,
            continuous_conditions=continuous_conditions,
        )
        # check attributes
        if continuous_conditions is None:
            assert schema.allows_grouping
        else:
            assert not schema.allows_grouping
        assert schema.categorical_conditions is not None
        if conditions is None:
            assert len(schema.categorical_conditions) == 0
        assert schema.categorical_reps is not None
        if categorical_reps is None:
            assert len(schema.categorical_reps) == 0
