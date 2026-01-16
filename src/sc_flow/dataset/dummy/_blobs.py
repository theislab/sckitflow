import scanpy as sc
from sklearn import datasets

from sc_flow.dataset.dummy._base import BaseDummyDataset


class BlobsDataset(BaseDummyDataset):
    """
    Generate isotropic Gaussian blobs for clustering.

    Parameters
    ----------
        random_state : int (default=42)
            Seed for random number generator to ensure reproducibility

        n_samples : int (default=1000)
            Total number of data points across all clusters

        n_features : int (default=2)
            Number of dimensions/features for each point

        centers : int (default=3)
            Number of distinct clusters to generate

        cluster_std : float (default=1.0)
            Standard deviation of clusters

        center_box : tuple of float (min,max) (default=(-10.0, 10.0))
            Bounding box for each cluster center

        shuffle : bool (default=True)
            Shuffles the samples
    """

    def __init__(
        self,
        random_state: int = 42,
        n_samples: int = 1000,
        n_features: int = 2,
        centers: int = 3,
        cluster_std: float = 1.0,
        center_box: tuple = (-10.0, 10.0),
        shuffle: bool = True,
    ):
        """Initialization"""
        super().__init__(
            random_state,
            n_samples,
            n_features,
            centers=centers,
            cluster_std=cluster_std,
            center_box=center_box,
            shuffle=shuffle,
        )

    def _get_dataset_name(self) -> str:
        """Return identifier for this dataset type."""
        return "blobs"

    def _validate_additional_parameters(self, **kwargs):
        """Validate common parameters to prevent errors"""
        centers = kwargs["centers"]
        cluster_std = kwargs["cluster_std"]
        center_box = kwargs["center_box"]
        shuffle = kwargs["shuffle"]

        if not isinstance(centers, int):
            raise TypeError(f"centers must be an integer, got {type(centers)}")
        if not isinstance(cluster_std, float):
            raise TypeError(f"cluster_std must be a float, got {type(cluster_std)}")
        if not isinstance(center_box, tuple):
            raise TypeError(f"center_box must be a tuple, got {type(center_box)}")
        if not isinstance(shuffle, bool):
            raise TypeError(f"shuffle must be a boolean, got {type(shuffle)}")

        if centers <= 0:
            raise ValueError(f"centers must be positive, got {centers}")
        if cluster_std <= 0:
            raise ValueError(f"cluster_std must be positive, got {cluster_std}")

    def _generate(self, random_state, n_samples, n_features, centers, cluster_std, center_box, shuffle) -> sc.AnnData:
        """
        Generate blob-shaped clusters.

        Returns
        -------
        adata : sc.AnnData
            Dataset with blob clusters, labels & metadata
        """
        # Step 1: Generate the blob data using sklearn
        X, y = datasets.make_blobs(
            n_samples=n_samples,
            n_features=n_features,
            centers=centers,
            cluster_std=cluster_std,
            random_state=random_state,
            center_box=center_box,
            shuffle=shuffle,
        )

        # Step 2: Wrap in AnnData format with standardized structure
        adata = self._create_anndata(X, y)

        # Step 3: Add additional metadata specific to blobs
        adata.uns["dataset_info"]["centers"] = centers
        adata.uns["dataset_info"]["cluster_std"] = cluster_std
        adata.uns["dataset_info"]["center_box"] = center_box
        adata.uns["dataset_info"]["shuffle"] = shuffle

        return adata
