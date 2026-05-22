import numpy as np
import pandas as pd
import pytest

from sc_flow.data.containers._categorical import CategoricalData
from sc_flow.data.containers._coupling import CouplingData
from sc_flow.data.containers._distribution import DistributionData
from sc_flow.data.containers._mixed_type import MixedTypeData
from sc_flow.data.containers._state import StateData
from sc_flow.preprocessing._state_data_preproc import StatePreprocessing
from sc_flow.preprocessing.transforms._base import BaseTransform, TransformParams


# -----------------------------------------------------------------------------
# Dummy transform for testing
# -----------------------------------------------------------------------------
class DummyTransform(BaseTransform):
    """Simple transform that multiplies by 2 and tracks fitted state."""

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
# Helper to create DistributionData (as provided)
# -----------------------------------------------------------------------------
def make_distribution(n=10):
    X = np.arange(n).reshape(-1, 1)
    df = pd.DataFrame({"grp": np.arange(n)})
    cat = CategoricalData.from_pandas(ann_df=df)
    state_data = StateData(X=X)

    response_data = MixedTypeData(categorical_covariates=cat)
    condition_data = MixedTypeData(categorical_covariates=cat)
    source_coupling = CouplingData.init_from_state_data(state_data)
    target_coupling = CouplingData.init_from_state_data(state_data)

    return DistributionData(
        state_data=state_data,
        response_data=response_data,
        condition_data=condition_data,
        groups_data=cat,
        source_coupling_data=source_coupling,
        target_coupling_data=target_coupling,
    )


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
class TestStatePreprocessing:
    def test_extract_data(self):
        distr = make_distribution(n=5)
        pre = StatePreprocessing()
        extracted = pre._extract_data(distr)
        np.testing.assert_array_equal(extracted, distr.state_data.X)

    def test_write_data(self):
        distr = make_distribution(n=5)
        pre = StatePreprocessing()
        new_X = np.random.randn(5, 1)
        new_distr = pre._write_data(new_X, distr)
        assert new_distr is not distr  # should return new object
        np.testing.assert_array_equal(new_distr.state_data.X, new_X)
        # Check other fields are unchanged (identity)
        assert new_distr.response_data is distr.response_data
        assert new_distr.condition_data is distr.condition_data
        assert new_distr.groups_data is distr.groups_data
        assert new_distr.source_coupling_data is distr.source_coupling_data
        assert new_distr.target_coupling_data is distr.target_coupling_data

    def test_full_pipeline_with_transform_only(self):
        distr = make_distribution(n=5)
        transform = DummyTransform()
        pre = StatePreprocessing(transform=transform)
        pre.fit(distr)
        transformed = pre.transform(distr)
        expected = distr.state_data.X * 2
        np.testing.assert_array_equal(transformed.state_data.X, expected)
        inverse = pre.inverse_transform(transformed)
        np.testing.assert_array_almost_equal(inverse.state_data.X, distr.state_data.X)

    def test_pipeline_with_encoder_only(self):
        distr = make_distribution(n=5)

        class DummyEncoder:
            def __call__(self, X):
                return X + 1

        encoder = DummyEncoder()
        pre = StatePreprocessing(encoder_context=encoder)
        pre.fit(distr)
        transformed = pre.transform(distr)
        expected = distr.state_data.X + 1
        np.testing.assert_array_equal(transformed.state_data.X, expected)

    def test_pipeline_with_transform_and_encoder(self):
        distr = make_distribution(n=5)
        transform = DummyTransform()

        class DummyEncoder:
            def __call__(self, X):
                return X + 1

        encoder = DummyEncoder()
        pre = StatePreprocessing(transform=transform, encoder_context=encoder)
        pre.fit(distr)
        transformed = pre.transform(distr)
        expected = distr.state_data.X * 2 + 1
        np.testing.assert_array_equal(transformed.state_data.X, expected)

    def test_pipeline_with_decoder_only(self):
        distr = make_distribution(n=5)

        class DummyDecoder:
            def __call__(self, X):
                return X - 1

        decoder = DummyDecoder()
        pre = StatePreprocessing(decoder_context=decoder)
        pre.fit(distr)
        inverse = pre.inverse_transform(distr)
        expected = distr.state_data.X - 1
        np.testing.assert_array_equal(inverse.state_data.X, expected)

    def test_pipeline_with_encoder_and_decoder(self):
        distr = make_distribution(n=5)
        transform = DummyTransform()

        class DummyEncoder:
            def __call__(self, X):
                return X + 1

        class DummyDecoder:
            def __call__(self, X):
                return X - 1

        encoder = DummyEncoder()
        decoder = DummyDecoder()
        pre = StatePreprocessing(transform=transform, encoder_context=encoder, decoder_context=decoder)
        pre.fit(distr)
        transformed = pre.transform(distr)
        inverse = pre.inverse_transform(transformed)
        np.testing.assert_array_almost_equal(inverse.state_data.X, distr.state_data.X)

    def test_fitted_property(self):
        distr = make_distribution(n=5)
        pre = StatePreprocessing()
        assert not pre.is_fitted
        pre.fit(distr)
        assert pre.is_fitted

    def test_transform_without_fit_raises(self):
        distr = make_distribution(n=5)
        pre = StatePreprocessing()
        with pytest.raises(RuntimeError, match="not fitted"):
            pre.transform(distr)
