import numpy as np
import pytest
import scanpy as sc

from sc_flow.data.preprocessing import PCAPreprocessor


class TestPCAPreprocessor:
    """Tests for PCA dimensionality reduction."""

    def test_fit_transform_basic(self):
        """Test basic fit and transform."""
        np.random.seed(42)
        X = np.random.randn(100, 50)

        pca = PCAPreprocessor(n_components=10)
        X_transformed = pca.fit_transform(X)

        # Check output shape
        assert X_transformed.shape == (100, 10)
        assert pca.is_fitted

    def test_fitted_parameters(self):
        """Test that parameters are correctly stored."""
        np.random.seed(42)
        X = np.random.randn(50, 20)

        pca = PCAPreprocessor(n_components=5)
        pca.fit(X)

        # Check stored parameters
        assert pca.is_fitted
        assert pca.params.n_features == 20
        assert pca.params.n_components == 5
        assert pca.params.mean.shape == (20,)
        assert pca.params.components.shape == (5, 20)
        assert pca.params.explained_variance.shape == (5,)

    def test_transform_with_new_data(self):
        """Test transform on new data."""
        np.random.seed(42)
        X_train = np.random.randn(100, 30)
        X_test = np.random.randn(20, 30)

        pca = PCAPreprocessor(n_components=10)
        pca.fit(X_train)
        X_test_transformed = pca.transform(X_test)

        assert X_test_transformed.shape == (20, 10)

    def test_inverse_transform(self):
        """Test inverse transform (lossy reconstruction)."""
        np.random.seed(42)
        X = np.random.randn(50, 20)

        pca = PCAPreprocessor(n_components=10)
        X_transformed = pca.fit_transform(X)
        X_reconstructed = pca.inverse_transform(X_transformed)

        # Shape should match original
        assert X_reconstructed.shape == X.shape

        # Reconstruction should be approximate (lossy)
        error = np.mean((X - X_reconstructed) ** 2)
        assert error > 0  # Some information lost
        assert error < 1.0  # But not too much

    def test_full_components_lossless(self):
        """Test that keeping all components gives lossless reconstruction."""
        np.random.seed(42)
        X = np.random.randn(50, 20)

        pca = PCAPreprocessor(n_components=20)  # Keep all
        X_transformed = pca.fit_transform(X)
        X_reconstructed = pca.inverse_transform(X_transformed)

        # Should be nearly lossless
        assert np.allclose(X, X_reconstructed, atol=1e-6)

    def test_explained_variance_ratio(self):
        """Test explained variance ratio."""
        np.random.seed(42)
        X = np.random.randn(100, 30)

        pca = PCAPreprocessor(n_components=10)
        pca.fit(X)

        var_ratio = pca.explained_variance_ratio()

        # Check properties
        assert len(var_ratio) == 10
        assert np.all(var_ratio >= 0)
        assert np.all(var_ratio <= 1)
        assert np.isclose(var_ratio.sum(), 1.0)
        # First PC should explain most variance
        assert var_ratio[0] == var_ratio.max()

    def test_not_fitted_error(self):
        """Test error when transform called before fit."""
        pca = PCAPreprocessor(n_components=5)
        X = np.random.randn(10, 20)

        with pytest.raises(RuntimeError, match="not been fitted"):
            pca.transform(X)

    def test_wrong_features_error(self):
        """Test error when n_features doesn't match."""
        X_train = np.random.randn(50, 30)
        X_test = np.random.randn(20, 40)

        pca = PCAPreprocessor(n_components=10)
        pca.fit(X_train)

        with pytest.raises(ValueError, match="features"):
            pca.transform(X_test)

    def test_manual_vs_ppca_transform(self):
        """Test manual PCA computation matches PCAPreprocessor output."""
        np.random.seed(42)
        X = np.random.randn(100, 50)

        # Manual PCA
        X_mean = np.array(np.mean(X, axis=0))
        X_centered = np.array(X - X_mean)
        X_pca, _, _, _ = sc.pp.pca(X_centered, zero_center=False, return_info=True, n_comps=30)

        # Using PCAPreprocessor
        pca = PCAPreprocessor(n_components=30)
        X_ppca_transformed = pca.fit_transform(X)

        # Compare
        assert np.allclose(X_pca, X_ppca_transformed, atol=1e-6)

    # def test_multiple_states(self):
    #     """Test fitting multiple times creates independent states."""
    #     np.random.seed(42)
    #     X1 = np.random.randn(50, 20)
    #     X2 = np.random.randn(50, 20) + 10  # Different distribution

    #     pca = PCAPreprocessor(n_components=5)

    #     # Fit on dataset 1
    #     params1 = pca.fit(X1)
    #     mean1 = params1.params.mean.copy()

    #     # Fit on dataset 2
    #     params2 = pca.fit(X2)
    #     mean2 = params2.params.mean.copy()

    #     # Means should be different
    #     assert not np.allclose(mean1, mean2)

    #     # Can use either params
    #     X_test = np.random.randn(10, 20)
    #     X_transformed1 = pca.transform(X_test, params=params1)
    #     X_transformed2 = pca.transform(X_test, params=params2)

    #     assert X_transformed1.shape == (10, 5)
    #     assert X_transformed2.shape == (10, 5)
    #     assert not np.allclose(X_transformed1, X_transformed2)


class TestParameterAccess:
    def test_pca_parameter_access(self):
        """Test accessing PCA parameters."""
        np.random.seed(42)
        X = np.random.randn(50, 20)

        pca = PCAPreprocessor(n_components=5)
        pca.fit(X)

        # Access via params
        assert pca.params.mean is not None
        assert pca.params.components is not None
        assert pca.params.explained_variance is not None
        assert pca.params.n_components == 5
        assert pca.params.n_features == 20


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
