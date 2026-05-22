import numpy as np
import pytest
from scipy.sparse import csr_matrix

pytest.importorskip("sc_flow.data.preprocessing")

from sc_flow.data.preprocessing import ZScoreTransform


class TestZScoreTransform:
    """Tests for Z-score normalization."""

    def test_fit_transform_basic(self):
        """Test basic fit and transform."""
        X = np.array([[1, 10], [2, 20], [3, 30], [4, 40], [5, 50]], dtype=float)

        zscore = ZScoreTransform()
        X_transformed = zscore.fit_transform(X)

        # Check transformed data has mean ~0 and std ~1
        assert np.allclose(X_transformed.mean(axis=0), 0, atol=1e-10)
        assert np.allclose(X_transformed.std(axis=0), 1, atol=1e-10)

    def test_fitted_parameters(self):
        """Test that parameters are correctly stored."""
        X = np.array([[1, 10], [2, 20], [3, 30]], dtype=float)

        zscore = ZScoreTransform()
        zscore.fit(X)

        # Check stored parameters
        assert zscore.is_fitted
        assert zscore.params.n_features == 2
        assert np.allclose(zscore.params.mean, [2.0, 20.0])
        assert len(zscore.params.std) == 2

    def test_transform_with_new_data(self):
        """Test transform on new data using fitted parameters."""
        X_train = np.array([[1, 10], [2, 20], [3, 30]], dtype=float)
        X_test = np.array([[2, 20], [4, 40]], dtype=float)

        zscore = ZScoreTransform()
        zscore.fit(X_train)
        X_test_transformed = zscore.transform(X_test)

        # Check shape preserved
        assert X_test_transformed.shape == X_test.shape

        # Test specific values
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        expected = (X_test - mean) / std
        assert np.allclose(X_test_transformed, expected)

    def test_inverse_transform(self):
        """Test that inverse transform recovers original data."""
        X = np.array([[1, 10], [2, 20], [3, 30], [4, 40]], dtype=float)

        zscore = ZScoreTransform()
        X_transformed = zscore.fit_transform(X)
        X_reconstructed = zscore.inverse_transform(X_transformed)

        # Should recover original data exactly
        assert np.allclose(X, X_reconstructed)

    def test_not_fitted_error(self):
        """Test error when transform called before fit."""
        zscore = ZScoreTransform()
        X = np.array([[1, 2], [3, 4]])

        with pytest.raises(RuntimeError, match="not been fitted"):
            zscore.transform(X)

    def test_wrong_features_error(self):
        """Test error when n_features doesn't match."""
        X_train = np.array([[1, 2, 3]], dtype=float)
        X_test = np.array([[1, 2]], dtype=float)

        zscore = ZScoreTransform()
        zscore.fit(X_train)

        with pytest.raises(ValueError, match="features"):
            zscore.transform(X_test)

    def test_epsilon_prevents_division_by_zero(self):
        """Test that epsilon prevents division by zero for constant features."""
        X = np.array([[1, 5], [1, 10], [1, 15]], dtype=float)  # First column constant

        zscore = ZScoreTransform(epsilon=1e-8)
        X_transformed = zscore.fit_transform(X)

        # Should not raise error or produce NaN/Inf
        assert not np.any(np.isnan(X_transformed))
        assert not np.any(np.isinf(X_transformed))

    def test_sparse_input(self):
        """Test that sparse matrices are converted properly."""
        X_dense = np.array([[1, 0, 2], [0, 3, 0], [4, 0, 5]], dtype=float)
        X_sparse = csr_matrix(X_dense)

        zscore = ZScoreTransform()
        X_transformed = zscore.fit_transform(X_sparse.toarray())

        assert X_transformed.shape == X_dense.shape


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_sample(self):
        """Test with single sample."""
        X = np.array([[1, 2, 3]], dtype=float)

        zscore = ZScoreTransform()
        X_transformed = zscore.fit_transform(X)

        # Should not crash
        assert X_transformed.shape == (1, 3)

    def test_single_feature(self):
        """Test with single feature."""
        X = np.array([[1], [2], [3], [4]], dtype=float)

        zscore = ZScoreTransform()
        X_transformed = zscore.fit_transform(X)

        assert X_transformed.shape == (4, 1)

    def test_invalid_input_not_2d(self):
        """Test error for non-2D input."""
        X = np.array([1, 2, 3, 4])

        zscore = ZScoreTransform()

        with pytest.raises(ValueError, match="2D array"):
            zscore.fit(X)

    def test_invalid_input_type(self):
        """Test error for non-numpy input."""
        X = [[1, 2], [3, 4]]  # List, not numpy array

        zscore = ZScoreTransform()

        with pytest.raises(TypeError, match="numpy array"):
            zscore.fit(X)

    def test_empty_array(self):
        """Test error for empty array."""
        X = np.array([]).reshape(0, 5)

        zscore = ZScoreTransform()

        with pytest.raises(ValueError, match="at least one sample"):
            zscore.fit(X)


class TestParameterAccess:
    """Test accessing fitted parameters."""

    def test_zscore_parameter_access(self):
        """Test accessing Z-score parameters."""
        X = np.array([[1, 10], [2, 20], [3, 30]], dtype=float)

        zscore = ZScoreTransform()
        zscore.fit(X)

        # Access via params
        assert zscore.params.mean is not None
        assert zscore.params.std is not None
        assert zscore.params.n_features == 2

        # Check values
        assert np.allclose(zscore.params.mean, [2.0, 20.0])


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
