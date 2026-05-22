import numpy as np
import pytest

from sc_flow.data.containers import DistributionData, StateData
from sc_flow.preprocessing._base_preproc import BasePreprocessing
from sc_flow.preprocessing.transforms._base import BaseTransform, TransformParams


# -----------------------------------------------------------------------------
# Dummy transform and contexts for testing
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


class DummyEncoder:
    def __call__(self, X):
        return X + 1


class DummyDecoder:
    def __call__(self, X):
        return X - 1


# -----------------------------------------------------------------------------
# Concrete subclass for testing (extracts state data)
# -----------------------------------------------------------------------------
class DummyPreprocessing(BasePreprocessing):
    def _extract_data(self, data: DistributionData) -> np.ndarray:
        return data.state_data.X

    def _write_data(self, X: np.ndarray, data: DistributionData) -> DistributionData:
        new_state = StateData(X)
        return DistributionData(
            state_data=new_state,
            response_data=data.response_data,
            condition_data=data.condition_data,
            groups_data=data.groups_data,
            source_coupling_data=data.source_coupling_data,
            target_coupling_data=data.target_coupling_data,
        )


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture
def sample_data():
    X = np.random.randn(10, 5)
    state = StateData(X)
    return DistributionData(state_data=state)


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
class TestBasePreprocessing:
    def test_init_default(self):
        pre = DummyPreprocessing()
        assert pre._transform is None
        assert pre._encoder_context is None
        assert pre._decoder_context is None
        assert not pre.is_fitted

    def test_init_with_transform(self):
        transform = DummyTransform()
        pre = DummyPreprocessing(transform=transform)
        assert pre._transform is transform
        assert not pre.is_fitted

    def test_fit_calls_transform_fit(self, sample_data):
        transform = DummyTransform()
        pre = DummyPreprocessing(transform=transform)
        pre.fit(sample_data)
        assert transform._fit_called
        assert pre.is_fitted

    def test_transform_without_fit_raises(self, sample_data):
        pre = DummyPreprocessing()
        with pytest.raises(RuntimeError, match="not fitted"):
            pre.transform(sample_data)

    def test_transform_with_transform_only(self, sample_data):
        transform = DummyTransform()
        pre = DummyPreprocessing(transform=transform)
        pre.fit(sample_data)
        transformed = pre.transform(sample_data)
        # Expected: original X * 2
        expected = sample_data.state_data.X * 2
        np.testing.assert_array_equal(transformed.state_data.X, expected)

    def test_transform_with_encoder_only(self, sample_data):
        encoder = DummyEncoder()
        pre = DummyPreprocessing(encoder_context=encoder)
        # No transform, but we still need to "fit" (does nothing)
        pre.fit(sample_data)
        transformed = pre.transform(sample_data)
        expected = sample_data.state_data.X + 1
        np.testing.assert_array_equal(transformed.state_data.X, expected)

    def test_transform_with_transform_and_encoder(self, sample_data):
        transform = DummyTransform()
        encoder = DummyEncoder()
        pre = DummyPreprocessing(transform=transform, encoder_context=encoder)
        pre.fit(sample_data)
        transformed = pre.transform(sample_data)
        # First multiply by 2, then add 1
        expected = sample_data.state_data.X * 2 + 1
        np.testing.assert_array_equal(transformed.state_data.X, expected)

    def test_inverse_transform_without_fit_raises(self, sample_data):
        pre = DummyPreprocessing()
        with pytest.raises(RuntimeError, match="not fitted"):
            pre.inverse_transform(sample_data)

    def test_inverse_transform_only_transform(self, sample_data):
        transform = DummyTransform()
        pre = DummyPreprocessing(transform=transform)
        pre.fit(sample_data)
        # First transform to get transformed data, then inverse
        transformed = pre.transform(sample_data)
        inverse = pre.inverse_transform(transformed)
        np.testing.assert_array_equal(inverse.state_data.X, sample_data.state_data.X)

    def test_inverse_transform_with_decoder_only(self, sample_data):
        decoder = DummyDecoder()
        pre = DummyPreprocessing(decoder_context=decoder)
        pre.fit(sample_data)
        # First transform (identity, no transform) then inverse (decoder)
        # To test, we need to pass data that has been encoded (but we don't have encoder)
        # Inverse_transform expects data that has been through forward.
        # Since forward (without encoder) is identity, inverse (decoder) should subtract 1.
        pre._encoder_context = None
        pre._decoder_context = decoder
        # We need to set the data to something that would have come from encoder.
        # Here we directly call inverse on original data (should subtract 1)
        inverse = pre.inverse_transform(sample_data)
        expected = sample_data.state_data.X - 1
        np.testing.assert_array_equal(inverse.state_data.X, expected)

    def test_inverse_transform_with_both_encoder_and_decoder(self, sample_data):
        transform = DummyTransform()
        encoder = DummyEncoder()
        decoder = DummyDecoder()
        pre = DummyPreprocessing(transform=transform, encoder_context=encoder, decoder_context=decoder)
        pre.fit(sample_data)
        transformed = pre.transform(sample_data)
        inverse = pre.inverse_transform(transformed)
        # Should recover original
        np.testing.assert_array_almost_equal(inverse.state_data.X, sample_data.state_data.X)

    def test_fitted_property_without_transform(self, sample_data):
        pre = DummyPreprocessing()
        assert not pre.is_fitted
        pre.fit(sample_data)
        assert pre.is_fitted

    def test_fitted_property_with_transform(self, sample_data):
        transform = DummyTransform()
        pre = DummyPreprocessing(transform=transform)
        assert not pre.is_fitted
        pre.fit(sample_data)
        assert pre.is_fitted
