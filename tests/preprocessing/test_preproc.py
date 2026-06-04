import numpy as np
import pytest

from sc_flow.data.containers._distribution import DistributionData
from sc_flow.preprocessing._preproc import DataPreprocessor
from sc_flow.preprocessing.preproc_containers._condition_data_preproc import ConditionPreprocessing
from sc_flow.preprocessing.preproc_containers._state_data_preproc import StatePreprocessing
from sc_flow.preprocessing.transforms._base import BaseTransform, TransformParams


# -----------------------------------------------------------------------------
# Dummy implementations for testing
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


class DummyStatePreprocessing(StatePreprocessing):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._fit_count = 0
        self._transform_count = 0
        self._inverse_count = 0
        self._unload_called = False

    def fit(self, data):
        self._fit_count += 1
        return super().fit(data)

    def transform(self, data):
        self._transform_count += 1
        return super().transform(data)

    def inverse_transform(self, data):
        self._inverse_count += 1
        return super().inverse_transform(data)

    def unload(self):
        self._unload_called = True


class DummyConditionPreprocessing(ConditionPreprocessing):
    def __init__(self, cov_key, **kwargs):
        super().__init__(cov_key, **kwargs)
        self._fit_count = 0
        self._transform_count = 0
        self._inverse_count = 0
        self._unload_called = False

    def fit(self, data):
        self._fit_count += 1
        return super().fit(data)

    def transform(self, data):
        self._transform_count += 1
        return super().transform(data)

    def inverse_transform(self, data):
        self._inverse_count += 1
        return super().inverse_transform(data)

    def unload(self):
        self._unload_called = True


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture
def sample_distribution():
    """Create a simple DistributionData with state and condition data."""
    from sc_flow.data._mixins import BatchMixin
    from sc_flow.data.containers._mixed_type import MixedTypeData
    from sc_flow.data.containers._state import StateData

    X_state = np.random.randn(10, 5)
    state_data = StateData(X_state)
    # Add a continuous condition covariate
    X_cont = np.random.randn(10, 3)
    cont_mapping = {"cov1": X_cont}
    continuous_covariates = BatchMixin(cont_mapping)
    condition_data = MixedTypeData(continuous_covariates=continuous_covariates)
    return DistributionData(
        state_data=state_data,
        condition_data=condition_data,
        response_data=None,
        groups_data=None,
        source_coupling_data=None,
        target_coupling_data=None,
    )


# -----------------------------------------------------------------------------
# Test suite
# -----------------------------------------------------------------------------
class TestDataPreprocessor:
    def test_init_default(self):
        preproc = DataPreprocessor()
        assert preproc._state_preproc is not None
        assert preproc._cond_preproc_dict == {}
        assert preproc.state_preproc is preproc._state_preproc

    def test_init_with_conditions_covariates(self):
        conditions = ["cov1", "cov2"]
        preproc = DataPreprocessor(conditions_covariates=conditions)
        assert set(preproc._cond_preproc_dict.keys()) == set(conditions)
        for cov in conditions:
            assert isinstance(preproc._cond_preproc_dict[cov], ConditionPreprocessing)
            assert preproc._cond_preproc_dict[cov]._cov_key == cov

    def test_init_with_condition_dicts_valid(self):
        conditions = ["cov1", "cov2"]
        transform_dict = {"cov1": DummyTransform(), "cov2": None}
        encoder_dict = {"cov1": DummyEncoder(), "cov2": None}
        decoder_dict = {"cov1": None, "cov2": DummyDecoder()}
        preproc = DataPreprocessor(
            conditions_covariates=conditions,
            condition_covariates_transform_dict=transform_dict,
            condition_covariates_encoder_context_dict=encoder_dict,
            condition_covariates_decoder_context_dict=decoder_dict,
        )
        assert preproc._cond_preproc_dict["cov1"]._transform is transform_dict["cov1"]
        assert preproc._cond_preproc_dict["cov1"]._encoder_context is encoder_dict["cov1"]
        assert preproc._cond_preproc_dict["cov2"]._transform is None
        assert preproc._cond_preproc_dict["cov2"]._encoder_context is None
        assert preproc._cond_preproc_dict["cov2"]._decoder_context is decoder_dict["cov2"]

    def test_fit_calls_state_and_condition_fit(self, sample_distribution):
        preproc = DataPreprocessor(
            conditions_covariates=["cov1"],
            state_transform=DummyTransform(),
        )
        # Replace internal preprocessors with dummy counters
        preproc._state_preproc = DummyStatePreprocessing()
        preproc._cond_preproc_dict = {
            "cov1": DummyConditionPreprocessing("cov1"),
        }
        preproc.fit(sample_distribution)
        assert preproc._state_preproc._fit_count == 1
        assert preproc._cond_preproc_dict["cov1"]._fit_count == 1

    def test_transform_calls_state_and_condition_transform(self, sample_distribution):
        preproc = DataPreprocessor(
            conditions_covariates=["cov1"],
            state_transform=DummyTransform(),
        )
        preproc._state_preproc = DummyStatePreprocessing()
        preproc._cond_preproc_dict = {
            "cov1": DummyConditionPreprocessing("cov1"),
        }
        # Need to fit first because transform requires fitted state
        preproc.fit(sample_distribution)
        # Reset counters
        preproc._state_preproc._transform_count = 0
        preproc._cond_preproc_dict["cov1"]._transform_count = 0
        transformed = preproc.transform(sample_distribution)
        assert preproc._state_preproc._transform_count == 1
        assert preproc._cond_preproc_dict["cov1"]._transform_count == 1
        # Check that the returned data is a DistributionData (type check)
        assert isinstance(transformed, DistributionData)

    def test_inverse_transform_calls_state_and_condition_inverse(self, sample_distribution):
        preproc = DataPreprocessor(
            conditions_covariates=["cov1"],
            state_transform=DummyTransform(),
        )
        preproc._state_preproc = DummyStatePreprocessing()
        preproc._cond_preproc_dict = {
            "cov1": DummyConditionPreprocessing("cov1"),
        }
        preproc.fit(sample_distribution)
        transformed = preproc.transform(sample_distribution)
        preproc._state_preproc._inverse_count = 0
        preproc._cond_preproc_dict["cov1"]._inverse_count = 0
        inverse = preproc.inverse_transform(transformed)
        assert preproc._state_preproc._inverse_count == 1
        assert preproc._cond_preproc_dict["cov1"]._inverse_count == 1
        assert isinstance(inverse, DistributionData)

    def test_unload_calls_state_and_condition_unload(self):
        preproc = DataPreprocessor(
            conditions_covariates=["cov1"],
        )
        preproc._state_preproc = DummyStatePreprocessing()
        preproc._cond_preproc_dict = {
            "cov1": DummyConditionPreprocessing("cov1"),
        }
        preproc.unload()
        assert preproc._state_preproc._unload_called
        assert preproc._cond_preproc_dict["cov1"]._unload_called

    def test_fit_does_nothing_if_no_preprocessing(self, sample_distribution):
        preproc = DataPreprocessor()
        # Should not raise
        preproc.fit(sample_distribution)
        # transform should just pass data through (no changes)
        transformed = preproc.transform(sample_distribution)
        np.testing.assert_array_equal(transformed.state_data.X, sample_distribution.state_data.X)

    def test_properties(self):
        preproc = DataPreprocessor()
        assert preproc.state_preproc is preproc._state_preproc
        assert preproc.cond_preproc is preproc._cond_preproc_dict

    def test_init_with_state_transform(self, sample_distribution):
        transform = DummyTransform()
        preproc = DataPreprocessor(state_transform=transform)
        preproc.fit(sample_distribution)
        assert preproc.state_preproc._transform is transform
        assert preproc.state_preproc.is_fitted

    def test_init_with_encoder_decoder_contexts(self, sample_distribution):
        encoder = DummyEncoder()
        decoder = DummyDecoder()
        preproc = DataPreprocessor(state_encoder_context=encoder, state_decoder_context=decoder)
        preproc.fit(sample_distribution)
        transformed = preproc.transform(sample_distribution)
        # Check that encoder was applied (state data +1)
        expected = sample_distribution.state_data.X + 1
        np.testing.assert_array_equal(transformed.state_data.X, expected)
        inverse = preproc.inverse_transform(transformed)
        np.testing.assert_array_almost_equal(inverse.state_data.X, sample_distribution.state_data.X)

    def test_init_with_no_conditions_covariates(self):
        preproc = DataPreprocessor(conditions_covariates=None)
        assert preproc._cond_preproc_dict == {}

    def test_cond_preproc_dict_creation_skips_when_conditions_none(self):
        preproc = DataPreprocessor(conditions_covariates=None)
        # Should not call check_sequence_query_against_reference
        assert preproc._cond_preproc_dict == {}
