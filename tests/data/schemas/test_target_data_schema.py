from collections.abc import Collection

import pytest
from anndata import AnnData

from sc_flow._types import TargetCovariatesEncodingId
from sc_flow.data._structures import TargetData
from sc_flow.data.schemas import TargetDataSchema

inval_key: str = "invalid_key"


class TestTargetDataSchema:
    @pytest.mark.parametrize(
        "categorical_covs_dict",
        [
            None,
            {"target": "one-hot"},
            {"target": "label"},
            {"target": "identity"},
            {"target": inval_key},
        ],
    )
    def test_init(
        self,
        categorical_covs_dict: dict[str, TargetCovariatesEncodingId] | None,
    ) -> None:
        """"""
        if categorical_covs_dict is not None and inval_key in categorical_covs_dict.values():
            with pytest.raises(ValueError, match="Covariate Encoder .* not available"):
                _schema = TargetDataSchema(categorical_covs_dict=categorical_covs_dict)
            return None
        _schema = TargetDataSchema(categorical_covs_dict=categorical_covs_dict)

    @pytest.mark.parametrize(
        "categorical_covs_dict",
        [
            None,
            {"target": "one-hot"},
            {"target": "label"},
            {"target": "identity"},
            {inval_key: "identity"},
        ],
    )
    @pytest.mark.parametrize("continuous_covs", [None, ["target_variable"], [inval_key]])
    def test_get_data(
        self,
        adata: AnnData,
        categorical_covs_dict: dict[str, TargetCovariatesEncodingId] | None,
        continuous_covs: Collection[str] | None,
    ) -> None:
        """"""
        if categorical_covs_dict is not None and inval_key in categorical_covs_dict.values():
            with pytest.raises(ValueError, match="Covariate Encoder .* not available"):
                schema = TargetDataSchema(categorical_covs_dict=categorical_covs_dict, continuous_covs=continuous_covs)
            return None
        schema = TargetDataSchema(categorical_covs_dict=categorical_covs_dict, continuous_covs=continuous_covs)
        data = schema.get_data(adata)
        assert isinstance(data, TargetData)
