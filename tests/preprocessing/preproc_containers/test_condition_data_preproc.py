import numpy as np
import pandas as pd
import pytest

from sckitflow.data._mixins import BatchMixin
from sckitflow.data.containers._categorical import CategoricalData
from sckitflow.data.containers._coupling import CouplingData
from sckitflow.data.containers._distribution import DistributionData
from sckitflow.data.containers._mixed_type import MixedTypeData
from sckitflow.data.containers._state import StateData
from sckitflow.preprocessing.preproc_containers._condition_data_preproc import ConditionPreprocessing
from sckitflow.preprocessing.transforms._base import BaseTransform, TransformParams


# -----------------------------------------------------------------------------
# Dummy transform (same as in StatePreprocessing tests)
# -----------------------------------------------------------------------------
class DummyTransform(BaseTransform):
    def __init__(self):
        super().__init__()
        self._params = TransformParams(is_fitted=False, n_features=None)
        self._fit_called = False

    def fit(self, X: np.ndarray):
        self._params = TransformParams(is_fitted=True, n_features=X.shape[1])
        self._fit_called = True
        return self

    def transform(self, X: np.ndarray):
        self._params._validate_fitted()
        return X * 2

    def inverse_transform(self, X: np.ndarray):
        self._params._validate_fitted()
        return X / 2


# -----------------------------------------------------------------------------
# Helper: create a DistributionData with a continuous condition covariate
# -----------------------------------------------------------------------------
def make_distribution_with_continuous_cov(n_obs=10, cov_key="X_cond", n_features=1):
    """Create a DistributionData with a continuous covariate in condition_data."""
    rng = np.random.default_rng(42)
    state_X = rng.random((n_obs, 3))
    state = StateData(X=state_X)

    # simple categorical data
    df = pd.DataFrame({"group": np.arange(n_obs)})
    cat_data = CategoricalData.from_pandas(ann_df=df)

    # continuous covariates: one key with array of shape (n_obs, n_features)
    cont_data = rng.random((n_obs, n_features))
    continuous_covariates = BatchMixin({cov_key: cont_data})

    condition_data = MixedTypeData(
        categorical_covariates=cat_data,
        continuous_covariates=continuous_covariates,
    )

    # response, groups, coupling are just placeholders
    response_data = MixedTypeData(categorical_covariates=cat_data)
    groups_data = cat_data
    source_coupling = CouplingData.init_from_state_data(state)
    target_coupling = CouplingData.init_from_state_data(state)

    return DistributionData(
        state_data=state,
        response_data=response_data,
        condition_data=condition_data,
        groups_data=groups_data,
        source_coupling_data=source_coupling,
        target_coupling_data=target_coupling,
    )


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
class TestConditionPreprocessing:
    def test_extract_data(self):
        cov_key = "X_cond"
        distr = make_distribution_with_continuous_cov(n_obs=5, cov_key=cov_key)
        pre = ConditionPreprocessing(cov_key=cov_key)
        extracted = pre._extract_data(distr)
        expected = distr.condition_data.continuous_covariates[cov_key]
        np.testing.assert_array_equal(extracted, expected)

    def test_extract_data_raises_on_missing_key(self):
        distr = make_distribution_with_continuous_cov(cov_key="X_cond")
        pre = ConditionPreprocessing(cov_key="nonexistent")
        with pytest.raises(KeyError, match="nonexistent"):
            pre._extract_data(distr)

    def test_extract_data_raises_if_no_continuous_covariates(self):
        # create a distribution without any continuous covariates
        df = pd.DataFrame({"group": [0, 1]})
        cat = CategoricalData.from_pandas(ann_df=df)
        state = StateData(X=np.random.randn(2, 2))
        distr = DistributionData(
            state_data=state,
            condition_data=MixedTypeData(categorical_covariates=cat, continuous_covariates=None),
        )
        pre = ConditionPreprocessing(cov_key="X_cond")
        with pytest.raises(ValueError, match="No continuous covariates"):
            pre._extract_data(distr)

    def test_write_data(self):
        cov_key = "X_cond"
        distr = make_distribution_with_continuous_cov(n_obs=5, cov_key=cov_key)
        pre = ConditionPreprocessing(cov_key=cov_key)
        new_X = np.random.randn(5, 1)
        new_distr = pre._write_data(new_X, distr)

        # The returned DistributionData must be a different object
        assert new_distr is not distr
        # The continuous covariate should have been updated
        np.testing.assert_array_equal(new_distr.condition_data.continuous_covariates[cov_key], new_X)
        # Other fields unchanged
        assert new_distr.state_data is distr.state_data
        assert new_distr.response_data is distr.response_data
        assert new_distr.groups_data is distr.groups_data
        assert new_distr.source_coupling_data is distr.source_coupling_data
        assert new_distr.target_coupling_data is distr.target_coupling_data

    def test_full_pipeline_with_transform_only(self):
        cov_key = "X_cond"
        distr = make_distribution_with_continuous_cov(n_obs=5, cov_key=cov_key)
        transform = DummyTransform()
        pre = ConditionPreprocessing(cov_key=cov_key, transform=transform)
        pre.fit(distr)
        transformed = pre.transform(distr)
        expected = distr.condition_data.continuous_covariates[cov_key] * 2
        np.testing.assert_array_equal(transformed.condition_data.continuous_covariates[cov_key], expected)
        inverse = pre.inverse_transform(transformed)
        np.testing.assert_array_almost_equal(
            inverse.condition_data.continuous_covariates[cov_key],
            distr.condition_data.continuous_covariates[cov_key],
        )

    def test_pipeline_with_encoder_only(self):
        cov_key = "X_cond"
        distr = make_distribution_with_continuous_cov(n_obs=5, cov_key=cov_key)

        class DummyEncoder:
            def __call__(self, X):
                return X + 1

        encoder = DummyEncoder()
        pre = ConditionPreprocessing(cov_key=cov_key, encoder_context=encoder)
        pre.fit(distr)
        transformed = pre.transform(distr)
        expected = distr.condition_data.continuous_covariates[cov_key] + 1
        np.testing.assert_array_equal(transformed.condition_data.continuous_covariates[cov_key], expected)

    def test_pipeline_with_transform_and_encoder(self):
        cov_key = "X_cond"
        distr = make_distribution_with_continuous_cov(n_obs=5, cov_key=cov_key)
        transform = DummyTransform()

        class DummyEncoder:
            def __call__(self, X):
                return X + 1

        encoder = DummyEncoder()
        pre = ConditionPreprocessing(cov_key=cov_key, transform=transform, encoder_context=encoder)
        pre.fit(distr)
        transformed = pre.transform(distr)
        expected = distr.condition_data.continuous_covariates[cov_key] * 2 + 1
        np.testing.assert_array_equal(transformed.condition_data.continuous_covariates[cov_key], expected)

    def test_pipeline_with_decoder_only(self):
        cov_key = "X_cond"
        distr = make_distribution_with_continuous_cov(n_obs=5, cov_key=cov_key)

        class DummyDecoder:
            def __call__(self, X):
                return X - 1

        decoder = DummyDecoder()
        pre = ConditionPreprocessing(cov_key=cov_key, decoder_context=decoder)
        pre.fit(distr)
        inverse = pre.inverse_transform(distr)
        expected = distr.condition_data.continuous_covariates[cov_key] - 1
        np.testing.assert_array_equal(inverse.condition_data.continuous_covariates[cov_key], expected)

    def test_pipeline_with_encoder_and_decoder(self):
        cov_key = "X_cond"
        distr = make_distribution_with_continuous_cov(n_obs=5, cov_key=cov_key)
        transform = DummyTransform()

        class DummyEncoder:
            def __call__(self, X):
                return X + 1

        class DummyDecoder:
            def __call__(self, X):
                return X - 1

        encoder = DummyEncoder()
        decoder = DummyDecoder()
        pre = ConditionPreprocessing(
            cov_key=cov_key,
            transform=transform,
            encoder_context=encoder,
            decoder_context=decoder,
        )
        pre.fit(distr)
        transformed = pre.transform(distr)
        inverse = pre.inverse_transform(transformed)
        np.testing.assert_array_almost_equal(
            inverse.condition_data.continuous_covariates[cov_key],
            distr.condition_data.continuous_covariates[cov_key],
        )

    def test_fitted_property(self):
        cov_key = "X_cond"
        distr = make_distribution_with_continuous_cov(n_obs=5, cov_key=cov_key)
        pre = ConditionPreprocessing(cov_key=cov_key)
        assert not pre.is_fitted
        pre.fit(distr)
        assert pre.is_fitted

    def test_transform_without_fit_raises(self):
        cov_key = "X_cond"
        distr = make_distribution_with_continuous_cov(n_obs=5, cov_key=cov_key)
        pre = ConditionPreprocessing(cov_key=cov_key)
        with pytest.raises(RuntimeError, match="not fitted"):
            pre.transform(distr)

    def test_inverse_transform_without_fit_raises(self):
        cov_key = "X_cond"
        distr = make_distribution_with_continuous_cov(n_obs=5, cov_key=cov_key)
        pre = ConditionPreprocessing(cov_key=cov_key)
        with pytest.raises(RuntimeError, match="not fitted"):
            pre.inverse_transform(distr)
