import pytest

from sc_flow.dataset.toy_data import dummy


def test_blobs():
    """Test Blobs dataset with all parameters."""

    dummy_func = dummy()

    # Test 1: Basic generation
    adata = dummy_func.get("blobs", n_samples=100, random_state=42).adata
    assert adata.shape == (100, 2)
    assert "labels" in adata.obs

    # Test 2: Custom parameters
    adata = dummy_func.get(
        "blobs",
        n_samples=1000,
        centers=4,
        cluster_std=1.5,
        center_box=(-20.0, 15.0),
        shuffle=False,
        random_state=1,
        n_features=8,
    ).adata
    assert adata.shape == (1000, 8)
    assert adata.obs["labels"].nunique() == 4

    # Test 3: Metadata
    assert "dataset_info" in adata.uns
    assert adata.uns["dataset_info"]["name"] == "blobs"
    assert "centers" in adata.uns["dataset_info"]


def test_checkerboard():
    """Test Checkerboard dataset."""

    dummy_func = dummy()

    # Test 1: Basic generation
    adata = dummy_func.get("checkerboard", n_samples=100, random_state=42, n_features=10).adata
    assert adata.shape == (100, 10)
    assert "row_cluster" in adata.obsm
    assert "col_cluster" in adata.varm

    # Test 2: Custom grid size
    adata = dummy_func.get("checkerboard", n_samples=200, n_clusters=(4, 5), random_state=42).adata
    expected_clusters = 4 * 5
    assert adata.obsm["row_cluster"].T.shape == (expected_clusters, 200)
    assert adata.varm["col_cluster"].T.shape == (expected_clusters, 2)

    # Test 3: Metadata
    assert "dataset_info" in adata.uns
    assert adata.uns["dataset_info"]["name"] == "checkerboard"
    assert "n_clusters" in adata.uns["dataset_info"]


def test_circles():
    """Test Circles dataset"""

    dummy_func = dummy()

    # Test 1: Basic generation
    adata = dummy_func.get("circles", n_samples=800, factor=0.6, noise=0.1, random_state=2).adata
    assert adata.shape == (800, 2)
    assert "labels" in adata.obs
    assert adata.obs["labels"].nunique() == 2

    # Test 2: Metadata
    assert "dataset_info" in adata.uns
    assert adata.uns["dataset_info"]["name"] == "circles"


def test_moons():
    """Test Moons dataset."""

    dummy_func = dummy()

    # Test 1: Basic generation
    adata = dummy_func.get("moons", n_samples=1000, noise=0.05, shuffle=False, random_state=3).adata
    assert adata.shape == (1000, 2)
    assert "labels" in adata.obs
    assert adata.obs["labels"].nunique() == 2

    # Test 2: Metadata
    assert "dataset_info" in adata.uns
    assert adata.uns["dataset_info"]["name"] == "moons"


def test_s_curve():
    """Test S-Curve dataset."""

    dummy_func = dummy()

    # Test 1: Basic generation
    adata = dummy_func.get("s_curve", n_samples=15000, noise=0.9, random_state=60).adata
    assert adata.shape == (15000, 3)
    assert "labels" in adata.obs

    # Test 2: Metadata
    assert "dataset_info" in adata.uns
    assert adata.uns["dataset_info"]["name"] == "scurve"


def test_swiss_roll():
    """Test Swiss Roll dataset."""

    dummy_func = dummy()

    # Test 1: Basic generation
    adata = dummy_func.get("swiss_roll", n_samples=2000, noise=0.5, hole=True, random_state=73).adata
    assert adata.shape == (2000, 3)
    assert "labels" in adata.obs

    # Test 2: Metadata
    assert "dataset_info" in adata.uns
    assert adata.uns["dataset_info"]["name"] == "swissroll"


def test_unified_interface():
    dummy_func = dummy()

    # Test 1: list_datasets()
    datasets_dict = dummy_func.list_datasets()
    assert len(datasets_dict) == 6
    assert "blobs" in datasets_dict

    # Test 2: get_dataset_info()
    info = dummy_func.get_dataset_info("blobs")
    assert len(info.keys()) == 7


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
