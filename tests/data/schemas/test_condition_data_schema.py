from collections.abc import Collection

import pytest
from anndata import AnnData

from sc_flow.data._structures import ConditionData
from sc_flow.data.schemas import ConditionDataSchema

inval_key: str = "invalid_key"


class TestConditionDataSchema:
    def _is_valid_args(
        self,
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
            assert schema.allows_ot_coupling
        else:
            assert not schema.allows_ot_coupling
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

        data = schema.get_data(adata)
        assert isinstance(data, ConditionData)
        # condition reps
        if conditions is not None:
            for condition in conditions:
                # repr dictionary
                assert condition in data.condition_reps.repr_dict
                cond_repr_dict = data.condition_reps.repr_dict[condition]
                for v in cond_repr_dict.values():
                    emb_dim = uns_keys_to_nunique_prefix_and_dim[condition][-1]
                    expected_shape = (1, emb_dim)
                    assert v.shape == expected_shape
                # annotation df
                condition_cols = conditions[condition]
                for cond in condition_cols:
                    assert cond in data.condition_reps.ann_df.columns
        # condition covariates
        if conditions_covariates is not None:
            for condition in conditions_covariates:
                val = data.condition_covariates.mapping[condition]
                emb_dim = obsm_keys_to_dim[condition]
                expected_shape = (len(adata), emb_dim)
                assert val.shape == expected_shape
