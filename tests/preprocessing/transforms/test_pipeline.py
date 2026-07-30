import numpy as np
import pytest

pytest.importorskip("sckitflow.data.preprocessing")

from sckitflow.data.preprocessing import PCATransform, ZScoreTransform


class TestPreprocessingPipeline:
    """Tests for chaining preprocessors."""

    def test_zscore_then_pca(self):
        """Test Z-score followed by PCA."""
        np.random.seed(42)
        X_train = np.random.randn(100, 50) * np.arange(1, 51)  # Different scales
        X_test = np.random.randn(20, 50) * np.arange(1, 51)

        # Step 1: Z-score
        zscore = ZScoreTransform()
        X_train_norm = zscore.fit_transform(X_train)
        X_test_norm = zscore.transform(X_test)

        # Step 2: PCA
        pca = PCATransform(n_components=10)
        X_train_pca = pca.fit_transform(X_train_norm)
        X_test_pca = pca.transform(X_test_norm)

        # Check shapes
        assert X_train_pca.shape == (100, 10)
        assert X_test_pca.shape == (20, 10)

        # Inverse pipeline
        X_train_recon_norm = pca.inverse_transform(X_train_pca)
        X_train_recon = zscore.inverse_transform(X_train_recon_norm)

        assert X_train_recon.shape == X_train.shape

    def test_fit_transform_returns_tuple(self):
        """Test that PCA fit_transform returns data and params."""
        np.random.seed(42)
        X = np.random.randn(50, 20)

        pca = PCATransform(n_components=5)
        X_transformed = pca.fit_transform(X)
        params = pca.params

        assert X_transformed.shape == (50, 5)
        assert params.n_components == 5
        assert params.is_fitted
