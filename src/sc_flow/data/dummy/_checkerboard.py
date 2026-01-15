import scanpy as sc
from sklearn import datasets

from sc_flow.data.dummy._base import BaseDummyDataset


class CheckerboardDataset(BaseDummyDataset):
    """
    Generate an array with block checkerboard structure for biclustering.

    Parameters
    ----------
        random_state : int (default=42)
            Seed for random number generator to ensure reproducibility

        n_samples : int (default=1000)
            Number of data points to generate

        n_features : int (default=2)
            Number of dimensions

        n_clusters : tuple|int (default=3)
            Number of row & column clusters

        noise : float (default=0.0)
            Standard deviation of the gaussian noise

        minval : float (default=10.0)
            Minumum value of a bicluster

        maxval : float (default=100.0)
            Maximum value of a bicluster

        shuffle : bool (default=True)
            Shuffles the samples
    """

    def __init__(
        self,
        random_state: int = 42,
        n_samples: int = 1000,
        n_features: int = 2,
        n_clusters: int | tuple[int, int] = 3,
        noise: float = 0.0,
        minval: float = 10.0,
        maxval: float = 100.0,
        shuffle: bool = True,
    ):
        super().__init__(
            random_state,
            n_samples,
            n_features,
            n_clusters=n_clusters,
            noise=noise,
            minval=minval,
            maxval=maxval,
            shuffle=shuffle,
        )

    def _get_dataset_name(self) -> str:
        """Return identifier for this dataset type."""
        return "checkerboard"

    def _validate_additional_parameters(self, **kwargs):
        """Validate common parameters to prevent errors"""
        n_clusters = kwargs["n_clusters"]
        noise = kwargs["noise"]
        minval = kwargs["minval"]
        maxval = kwargs["maxval"]
        shuffle = kwargs["shuffle"]

        if not (isinstance(n_clusters, int) or isinstance(n_clusters, tuple)):
            raise TypeError(f"n_clusters must be an integer or tuple of integers, got {type(n_clusters)}")

        if isinstance(n_clusters, tuple):
            if len(n_clusters) != 2:
                raise ValueError(f"n_clusters tuple must have length 2, got {n_clusters}")
            if not all(isinstance(x, int) for x in n_clusters):
                raise TypeError(f"n_clusters tuple must contain integers, got {n_clusters}")

        if not isinstance(noise, float):
            raise TypeError(f"noise must be a float, got {type(noise)}")

        if not isinstance(minval, float):
            raise TypeError(f"minval must be a float, got {type(minval)}")

        if not isinstance(maxval, float):
            raise TypeError(f"maxval must be a float, got {type(maxval)}")

        if not isinstance(shuffle, bool):
            raise TypeError(f"shuffle must be a boolean, got {type(shuffle)}")

    def _generate(self, random_state, n_samples, n_features, n_clusters, noise, minval, maxval, shuffle) -> sc.AnnData:
        """
        Generate a checkerboard pattern of clusters.

        Returns
        -------
        adata : sc.AnnData
            Dataset with checkerboard pattern and cluster labels
        """
        # Step 1: Generate checkerboard data
        X, rows, cols = datasets.make_checkerboard(
            random_state=random_state,
            shape=(n_samples, n_features),
            n_clusters=n_clusters,
            noise=noise,
            minval=minval,
            maxval=maxval,
            shuffle=shuffle,
        )

        # Step 2: Wrap in AnnData format
        adata = self._create_anndata(X)
        adata.obsm["row_cluster"] = rows.T
        adata.varm["col_cluster"] = cols.T

        # Add checkerboard-specific metadata
        adata.uns["dataset_info"]["n_clusters"] = n_clusters
        adata.uns["dataset_info"]["noise"] = noise
        adata.uns["dataset_info"]["minval"] = minval
        adata.uns["dataset_info"]["maxval"] = maxval
        adata.uns["dataset_info"]["shuffle"] = shuffle

        return adata
