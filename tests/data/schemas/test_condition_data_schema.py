from collections.abc import Collection

import pytest
from anndata import AnnData

from sc_flow.core.data.containers import MixedTypeData
from sc_flow.core.data.schemas import ConditionDataSchema

from ..shared import verify_categorical_data, verify_mixin  # noqa

inval_key: str = "invalid_key"


class TestConditionDataSchema:
    @staticmethod
    def _is_valid_args(
        conditions: dict[str, Collection[str]],
        conditions_reps: dict[str, str],
    ) -> bool:
        if conditions is None and conditions_reps is None:
            return True
        elif conditions is not None and conditions_reps is not None:
            return set(conditions.keys()) == set(conditions_reps.keys())
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
        "conditions_reps",
        [
            None,
            {"drug": "drug"},
            {"drug": "drug", "ko": "ko"},
            {inval_key: "drug", "ko": "ko"},
        ],
    )
    @pytest.mark.parametrize(
        "conditions_covariates",
        [
            None,
            ["drugA_time", "drugA_dose"],
        ],
    )
    def test_init(
        self,
        conditions: dict[str, Collection[str]],
        conditions_reps: dict[str, str],
        conditions_covariates: Collection[str] | None,
    ) -> None:
        # failure at initialization
        is_valid_args = self._is_valid_args(conditions, conditions_reps)
        if (
            (conditions is not None and conditions_reps is None)
            or (conditions is None and conditions_reps is not None)
            or not is_valid_args
        ):
            with pytest.raises(ValueError):
                # with pytest.raises(ValueError, match="The following reference columns are missing"):
                schema = ConditionDataSchema(
                    conditions=conditions, conditions_reps=conditions_reps, conditions_covariates=conditions_covariates
                )
            return None
        # init schema
        schema = ConditionDataSchema(
            conditions=conditions, conditions_reps=conditions_reps, conditions_covariates=conditions_covariates
        )
        # check attributes
        if conditions_covariates is None:
            assert schema.allows_grouping
        else:
            assert not schema.allows_grouping
        assert schema.conditions is not None
        if conditions is None:
            assert len(schema.conditions) == 0
        assert schema.conditions_reps is not None
        if conditions_reps is None:
            assert len(schema.conditions_reps) == 0

    @pytest.mark.parametrize(
        "conditions",
        [
            None,
            {
                "drug": ["drugA", "drugB"],
                "ko": ["koA", "koB"],
            },
        ],
    )
    @pytest.mark.parametrize(
        "conditions_reps",
        [
            None,
            {"drug": "drug"},
            {"drug": "drug", "ko": "ko"},
            {"drug": inval_key, "ko": "ko"},
        ],
    )
    @pytest.mark.parametrize(
        "conditions_covariates",
        [
            None,
            ["drugA_time", "drugA_dose"],
            [inval_key],
        ],
    )
    def test_get_data(
        self,
        adata: AnnData,
        uns_keys_to_nunique_prefix_and_dim: dict[str, tuple[int, str, int]],
        obsm_keys_to_dim: dict[str, int],
        conditions: dict[str, Collection[str]] | None,
        conditions_reps: dict[str, str] | None,
        conditions_covariates: Collection[str] | None,
    ) -> None:
        is_valid_args = self._is_valid_args(conditions, conditions_reps)
        if not is_valid_args:
            pytest.skip()
        schema = ConditionDataSchema(
            conditions=conditions,
            conditions_reps=conditions_reps,
            conditions_covariates=conditions_covariates,
        )
        if conditions_covariates is not None and inval_key in conditions_covariates:
            with pytest.raises(KeyError, match=r"Key .* not found in adata"):
                data = schema.get_data(adata)
            return None
        if conditions is not None and inval_key in [e for val in conditions.values() for e in val]:
            with pytest.raises(KeyError, match=r"Key .* not found in adata"):
                data = schema.get_data(adata)
            return None
        if conditions_reps is not None and inval_key in conditions_reps.values():
            with pytest.raises(KeyError, match=r"Key .* not found in adata"):
                data = schema.get_data(adata)
            return None

        # get data
        data = schema.get_data(adata)
        if conditions is None and conditions_covariates is None:
            assert data is None
            return
        else:
            assert isinstance(data, MixedTypeData)

        # test condition reps
        if conditions is not None:
            expected_df_cols = [col for val in conditions.values() for col in val]
            verify_categorical_data(
                data.categorical_covariates,
                expected_df_cols,
                uns_keys_to_nunique_prefix_and_dim,
                conditions_reps=conditions_reps,
            )
        else:
            assert data.categorical_covariates is None

        # test condition covariates
        N = len(adata)
        verify_mixin(data.continuous_covariates, N, obsm_keys_to_dim, conditions_covariates)


def test_conditions_encoding_explicit_one_hot() -> None:
    """A level may declare `conditions_encoding` (e.g. one-hot) instead of a rep — explicit, not inferred."""
    # rep + explicit one-hot level: valid
    ConditionDataSchema(
        conditions={"drug": ["drug1"], "ko": ["koA", "koB"]},
        conditions_reps={"drug": "drug"},
        conditions_encoding={"ko": "one-hot"},
    )
    # a level with neither a rep nor an encoding is rejected (explicit representation required)
    with pytest.raises(ValueError):
        ConditionDataSchema(conditions={"drug": ["drug1"]})
    # a level cannot be both rep'd and encoded
    with pytest.raises(ValueError):
        ConditionDataSchema(
            conditions={"drug": ["drug1"]}, conditions_reps={"drug": "drug"}, conditions_encoding={"drug": "one-hot"}
        )
