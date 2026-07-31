import numpy as np
import pytest
from anndata import AnnData

from sckitflow.dataset.toy_data import get_toy_dataset

# (registry name, `dataset_info["name"]`, expected number of features)
# The registry key and the recorded name differ for the two manifold datasets.
DATASETS = [
    ("blobs", "blobs", 2),
    ("checkerboard", "checkerboard", 2),
    ("circles", "circles", 2),
    ("moons", "moons", 2),
    ("s_curve", "scurve", 3),
    ("swiss_roll", "swissroll", 3),
]
DATASET_NAMES = [name for name, _, _ in DATASETS]


class TestGetToyDataset:
    """Behaviour shared by every dataset in the registry."""

    @pytest.mark.parametrize(("name", "info_name", "n_features"), DATASETS)
    def test_builds_anndata(self, name, info_name, n_features):
        dataset = get_toy_dataset(name, n_samples=50, random_state=0)

        assert dataset.dataset_name == info_name
        adata = dataset.adata
        assert isinstance(adata, AnnData)
        assert adata.shape == (50, n_features)
        assert np.isfinite(adata.X).all()

    @pytest.mark.parametrize(("name", "info_name", "n_features"), DATASETS)
    def test_records_metadata(self, name, info_name, n_features):
        info = get_toy_dataset(name, n_samples=50, random_state=7).adata.uns["dataset_info"]

        assert info["name"] == info_name
        assert info["random_state"] == 7

    @pytest.mark.parametrize("name", DATASET_NAMES)
    def test_random_state_is_reproducible(self, name):
        first = get_toy_dataset(name, n_samples=40, random_state=3).adata
        second = get_toy_dataset(name, n_samples=40, random_state=3).adata

        np.testing.assert_array_equal(first.X, second.X)

    @pytest.mark.parametrize("name", DATASET_NAMES)
    def test_random_state_changes_output(self, name):
        first = get_toy_dataset(name, n_samples=40, random_state=3).adata
        other = get_toy_dataset(name, n_samples=40, random_state=4).adata

        assert not np.array_equal(first.X, other.X)

    @pytest.mark.parametrize("name", DATASET_NAMES)
    def test_n_samples_is_respected(self, name):
        assert get_toy_dataset(name, n_samples=64, random_state=0).adata.n_obs == 64

    def test_unknown_dataset_name(self):
        with pytest.raises(ValueError, match="Unknown dataset name 'not_a_dataset'"):
            get_toy_dataset("not_a_dataset")


class TestLabels:
    """Labels live in ``obsm["Y"]`` as an ``(n_obs, 1)`` column."""

    # Categorical datasets store labels as strings, manifolds as floats.
    @pytest.mark.parametrize(("name", "n_classes"), [("blobs", 3), ("circles", 2), ("moons", 2)])
    def test_categorical_labels(self, name, n_classes):
        adata = get_toy_dataset(name, n_samples=60, random_state=0).adata

        labels = adata.obsm["Y"]
        assert labels.shape == (60, 1)
        assert labels.dtype.kind == "U"
        assert len(np.unique(labels)) == n_classes

    @pytest.mark.parametrize("name", ["s_curve", "swiss_roll"])
    def test_continuous_labels(self, name):
        adata = get_toy_dataset(name, n_samples=60, random_state=0).adata

        labels = adata.obsm["Y"]
        assert labels.shape == (60, 1)
        assert labels.dtype.kind == "f"
        # A manifold position, so it varies continuously rather than taking a few values.
        assert len(np.unique(labels)) > 2

    def test_checkerboard_has_no_label_column(self):
        # Checkerboard is a biclustering dataset; it carries row/col clusters instead.
        assert "Y" not in get_toy_dataset("checkerboard", n_samples=60, random_state=0).adata.obsm


class TestBlobs:
    def test_custom_shape_and_centers(self):
        adata = get_toy_dataset("blobs", n_samples=200, n_features=8, centers=4, random_state=1).adata

        assert adata.shape == (200, 8)
        assert len(np.unique(adata.obsm["Y"])) == 4

    def test_blob_specific_metadata(self):
        adata = get_toy_dataset(
            "blobs", n_samples=100, centers=3, cluster_std=1.5, center_box=(-20.0, 15.0), random_state=42
        ).adata

        info = adata.uns["dataset_info"]
        assert info["No of centers"] == 3
        assert info["cluster_std"] == 1.5
        assert info["center_box"] == (-20.0, 15.0)
        # One recorded coordinate per center, keyed by its index.
        assert sorted(info["centers"]) == ["0", "1", "2"]
        assert all(center.shape == (2,) for center in info["centers"].values())


class TestCheckerboard:
    def test_cluster_assignments(self):
        # `make_checkerboard` produces n_row_clusters * n_col_clusters biclusters.
        adata = get_toy_dataset("checkerboard", n_samples=200, n_features=10, n_clusters=(4, 5), random_state=42).adata

        assert adata.shape == (200, 10)
        assert adata.obsm["row_cluster"].shape == (200, 20)
        assert adata.varm["col_cluster"].shape == (10, 20)

    def test_checkerboard_specific_metadata(self):
        info = get_toy_dataset(
            "checkerboard", n_samples=100, n_clusters=(2, 3), minval=5.0, maxval=50.0, random_state=0
        ).adata.uns["dataset_info"]

        assert info["n_clusters"] == (2, 3)
        assert info["minval"] == 5.0
        assert info["maxval"] == 50.0

    def test_values_lie_within_configured_range(self):
        # With no noise the block values are drawn from [minval, maxval].
        adata = get_toy_dataset(
            "checkerboard", n_samples=100, n_features=10, noise=0.0, minval=10.0, maxval=100.0, random_state=0
        ).adata

        assert adata.X.min() >= 0.0
        assert adata.X.max() <= 100.0


class TestCirclesAndMoons:
    def test_circles_factor_controls_inner_radius(self):
        # `factor` is the ratio of the inner to the outer circle radius.
        adata = get_toy_dataset("circles", n_samples=200, factor=0.5, noise=0.0, random_state=2).adata

        radii = np.linalg.norm(adata.X, axis=1)
        inner, outer = np.unique(np.round(radii, 6))
        assert inner == pytest.approx(0.5, abs=1e-6)
        assert outer == pytest.approx(1.0, abs=1e-6)

    def test_moons_noise_perturbs_points(self):
        clean = get_toy_dataset("moons", n_samples=200, noise=0.0, random_state=3).adata
        noisy = get_toy_dataset("moons", n_samples=200, noise=0.2, random_state=3).adata

        assert not np.allclose(clean.X, noisy.X)
        # Same underlying moons, so the perturbation stays on the order of the noise scale.
        assert np.abs(clean.X - noisy.X).mean() < 1.0


class TestManifolds:
    def test_swiss_roll_hole(self):
        without_hole = get_toy_dataset("swiss_roll", n_samples=500, hole=False, random_state=0).adata
        with_hole = get_toy_dataset("swiss_roll", n_samples=500, hole=True, random_state=0).adata

        assert without_hole.shape == with_hole.shape == (500, 3)
        assert not np.array_equal(without_hole.X, with_hole.X)
        assert with_hole.uns["dataset_info"]["hole"] is True

    def test_s_curve_noise_recorded(self):
        info = get_toy_dataset("s_curve", n_samples=100, noise=0.9, random_state=60).adata.uns["dataset_info"]

        assert info["noise"] == 0.9
