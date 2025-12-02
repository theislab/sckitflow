from collections.abc import Collection

import pytest

from sc_flow.data.schemas import ConditionDataSchema

inval_key: str = "invalid_key"


class TestConditionDataSchema:
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
            {
                "drug": ["drugA", "drugB"],
                "ko": [inval_key],
            },
        ],
    )
    @pytest.mark.parametrize(
        "conditions_reps",
        [
            None,
            {"drug": "drug"},
            {"drug": "drug", "ko": "ko"},
            {inval_key: "drug", "ko": "ko"},
        ],
    )
    def test_init(
        self,
        conditions: dict[str, Collection[str]],
        conditions_reps: dict[str, str],
    ) -> None:
        # failure at initialization
        if conditions is not None and conditions_reps is not None:
            same_keys_flag = set(conditions.keys()) == set(conditions_reps.keys())
        if conditions is None and conditions_reps is None:
            same_keys_flag = True
        else:
            same_keys_flag = False
        if (
            (conditions is not None and conditions_reps is None)
            or (conditions is None and conditions_reps is not None)
            or not same_keys_flag
        ):
            with pytest.raises(ValueError):
                # with pytest.raises(ValueError, match="The following reference columns are missing"):
                schema = ConditionDataSchema(
                    conditions=conditions,
                    conditions_reps=conditions_reps,
                )
        # init schema
        schema = ConditionDataSchema(
            conditions=conditions,
            conditions_reps=conditions_reps,
        )
        # check attributes
        assert schema.conditions is not None
        assert schema.conditions_reps is not None
        assert schema.allows_ot_coupling
