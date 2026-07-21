from collections.abc import Collection

import pytest
from anndata import AnnData

from sc_flow._types import TargetCovariatesEncodingId
from sc_flow.core.data.containers import MixedTypeData
from sc_flow.core.data.schemas import ResponseDataSchema

from ..shared import verify_categorical_data, verify_mixin  # noqa

inval_key: str = "invalid_key"


class TestResponseDataSchema:
    @pytest.mark.parametrize(
        "categorical_covs_dict",
        [
            None,
            {"target": "one-hot"},
            {"target": "label"},
            {"target": "functional"},
            {"target": inval_key},
        ],
    )
    def test_init(
        self,
        categorical_covs_dict: dict[str, TargetCovariatesEncodingId] | None,
    ) -> None:
        """"""
        if categorical_covs_dict is not None and inval_key in categorical_covs_dict.values():
            with pytest.raises(
                ValueError, match=r"Encoder identifier .* for target covariate encoding is not supported"
            ):
                _schema = ResponseDataSchema(categorical_covs_dict=categorical_covs_dict)
            return None
        _schema = ResponseDataSchema(categorical_covs_dict=categorical_covs_dict)

    @pytest.mark.parametrize(
        "categorical_covs_dict",
        [
            None,
            {"target": "one-hot"},
            {"target": "label"},
            {"target": "functional"},
            {inval_key: "functional"},
        ],
    )
    @pytest.mark.parametrize("continuous_covs", [None, ["target_variable"], [inval_key]])
    def test_get_data(
        self,
        adata: AnnData,
        uns_keys_to_nunique_prefix_and_dim: dict[str, tuple[int, str, int]],
        obsm_keys_to_dim: dict[str, int],
        categorical_covs_dict: dict[str, TargetCovariatesEncodingId] | None,
        continuous_covs: Collection[str] | None,
    ) -> None:
        """"""
        schema = ResponseDataSchema(categorical_covs_dict=categorical_covs_dict, continuous_covs=continuous_covs)
        if (categorical_covs_dict is not None and inval_key in categorical_covs_dict) or (
            continuous_covs is not None and inval_key in continuous_covs
        ):
            with pytest.raises(KeyError, match=r"Key .* not found in adata"):
                data = schema.get_data(adata)
            return None
        data = schema.get_data(adata)
        if categorical_covs_dict is None and continuous_covs is None:
            assert data is None
            return
        else:
            assert isinstance(data, MixedTypeData)

        if categorical_covs_dict is not None:
            expected_df_cols = list(categorical_covs_dict.keys())
            verify_categorical_data(
                data.categorical_covariates,
                expected_df_cols,
                uns_keys_to_nunique_prefix_and_dim,
                groups_encoding=categorical_covs_dict,
            )
        else:
            assert data.categorical_covariates is None

        # test continuous covariates
        N = len(adata)
        verify_mixin(data.continuous_covariates, N, obsm_keys_to_dim, continuous_covs)
