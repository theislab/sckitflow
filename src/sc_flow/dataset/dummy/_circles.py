import scanpy as sc
from sklearn import datasets

from sc_flow.dataset.dummy._base import BaseDummyDataset


class CirclesDataset(BaseDummyDataset):
    """
    Generate two concentric circles dataset.

    Parameters
    ----------
    random_state : int, optional (default=42)
        Seed for random number generation to ensure reproducibility

    n_samples : int, optional (default=1000)
        Total number of samples to generate

    noise : float, optional (default=0.0)
        Standard deviation of Gaussian noise added to the data

    factor : float, optional (default=0.8)
        Scale factor between inner and outer circle (must be between 0 and 1)

    shuffle : bool, optional (default=True)
        Whether to shuffle the samples after generation
    """

    _dataset_name: str = "circles"

    def __init__(
        self,
        random_state: int = 42,
        n_samples: int = 1000,
        noise: float = 0.0,
        factor: float = 0.8,
        shuffle: bool = True,
    ):
        """Initialization"""
        super().__init__(random_state, n_samples, noise=noise, factor=factor, shuffle=shuffle)

    def _generate(self, random_state, n_samples, noise, factor, shuffle) -> sc.AnnData:
        """
        Generate concentric circles dataset.

        Returns
        -------
        adata : sc.AnnData
            Dataset with two concentric circles labeled as class 0 and 1
        """
        # Step 1: Generate circles using sklearn
        X, y = datasets.make_circles(
            n_samples=n_samples, noise=noise, factor=factor, random_state=random_state, shuffle=shuffle
        )

        # Step 2: Wrap in AnnData format
        adata = self._create_anndata(X, y, y_type="category")

        # Step 3: Add circles-specific metadata
        adata.uns["dataset_info"]["noise"] = noise
        adata.uns["dataset_info"]["factor"] = factor
        adata.uns["dataset_info"]["shuffle"] = shuffle

        return adata
