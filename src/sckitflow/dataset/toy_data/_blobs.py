from anndata import AnnData
from sklearn import datasets

from sckitflow.dataset.toy_data._base import BaseDummyDataset


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

    _dataset_name: str = "blobs"

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
            n_features=n_features,
            centers=centers,
            cluster_std=cluster_std,
            center_box=center_box,
            shuffle=shuffle,
        )

    def _generate(self, random_state, n_samples, n_features, centers, cluster_std, center_box, shuffle) -> AnnData:
        """
        Generate blob-shaped clusters.

        Returns
        -------
        adata : AnnData
            Dataset with blob clusters, labels & metadata
        """
        # Step 1: Generate the blob data using sklearn
        X, y, centers_array = datasets.make_blobs(
            n_samples=n_samples,
            n_features=n_features,
            centers=centers,
            cluster_std=cluster_std,
            random_state=random_state,
            center_box=center_box,
            shuffle=shuffle,
            return_centers=True,
        )

        # Step 2: Wrap in AnnData format with standardized structure
        adata = self._create_anndata(X, y, y_type="category")

        # Step 3: Add additional metadata specific to blobs
        adata.uns["dataset_info"]["No of centers"] = centers
        adata.uns["dataset_info"]["cluster_std"] = cluster_std
        adata.uns["dataset_info"]["center_box"] = center_box
        adata.uns["dataset_info"]["shuffle"] = shuffle
        adata.uns["dataset_info"]["random_state"] = random_state

        adata.uns["dataset_info"]["centers"] = {str(i): center for i, center in enumerate(centers_array)}

        return adata
