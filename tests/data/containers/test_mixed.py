import numpy as np
import pytest
from anndata import AnnData

from sc_flow.data._mixins import BatchMixin
from sc_flow.data.containers import CategoricalData, MixedTypeData


class TestMixedTypeData:
    @pytest.mark.parametrize("use_categorical_covariates", [True, False])
    @pytest.mark.parametrize("use_continuous_covariates", [True, False])
    def test_init(
        self,
        adata: AnnData,
        use_categorical_covariates: bool,
        use_continuous_covariates: bool,
    ) -> None:
        if not use_categorical_covariates and not use_continuous_covariates:
            with pytest.raises(ValueError, match="must contain at least one covariate container"):
                MixedTypeData()
            return

        categorical_data = CategoricalData.from_pandas(adata.obs) if use_categorical_covariates else None
        continuous_data = BatchMixin(adata.obsm) if use_continuous_covariates else None

        mixed_data = MixedTypeData(
            categorical_covariates=categorical_data,
            continuous_covariates=continuous_data,
        )

        assert mixed_data.categorical_covariates is categorical_data
        assert mixed_data.continuous_covariates is continuous_data
        assert len(mixed_data) == adata.n_obs

    def test_init_shape_mismatch(self, adata: AnnData) -> None:
        categorical_data = CategoricalData.from_pandas(adata.obs.iloc[:-1])
        continuous_data = BatchMixin(adata.obsm)

        with pytest.raises(ValueError, match="Shape mismatch"):
            MixedTypeData(
                categorical_covariates=categorical_data,
                continuous_covariates=continuous_data,
            )

    def test_getitem_slice(self, adata: AnnData) -> None:
        categorical_data = CategoricalData.from_pandas(adata.obs)
        continuous_data = BatchMixin(adata.obsm)

        mixed_data = MixedTypeData(
            categorical_covariates=categorical_data,
            continuous_covariates=continuous_data,
        )

        subset = mixed_data[:5]

        assert isinstance(subset, MixedTypeData)
        assert len(subset) == 5
        assert subset.categorical_covariates is not None
        assert subset.continuous_covariates is not None

    def test_getitem_array(self, adata: AnnData) -> None:
        categorical_data = CategoricalData.from_pandas(adata.obs)
        continuous_data = BatchMixin(adata.obsm)

        mixed_data = MixedTypeData(
            categorical_covariates=categorical_data,
            continuous_covariates=continuous_data,
        )

        idxs = np.array([0, 2, 4])
        subset = mixed_data[idxs]

        assert len(subset) == len(idxs)
