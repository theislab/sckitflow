import scanpy as sc
from sklearn import datasets

from sc_flow.dataset.dummy._base import BaseDummyDataset


class SCurveDataset(BaseDummyDataset):
    """
    Generate S-Curve dataset.

    Parameters
    ----------
    random_state : int, optional (default=42)
        Seed for random number generation to ensure reproducibility

    n_samples : int, optional (default=1000)
        Total number of samples to generate

    noise : float, optional (default=0.0)
        Standard deviation of Gaussian noise added to the data
    """

    _dataset_name: str = "scurve"

    def __init__(
        self,
        random_state: int = 42,
        n_samples: int = 1000,
        noise: float = 0.0,
    ):
        """Initialization"""
        super().__init__(random_state, n_samples, noise=noise)

    def _generate(self, random_state, n_samples, noise) -> sc.AnnData:
        """
        Generate S-Curve dataset.

        Returns
        -------
        adata : sc.AnnData
            Dataset with two interleaving half-circles labeled as class 0 and 1
        """
        # Step 1: Generate moons using sklearn
        X, y = datasets.make_s_curve(
            n_samples=n_samples,
            noise=noise,
            random_state=random_state,
        )

        # Step 2: Wrap in AnnData format
        adata = self._create_anndata(X, y, y_type=None)

        # Step 3: Add scurve-specific metadata
        adata.uns["dataset_info"]["noise"] = noise

        return adata
