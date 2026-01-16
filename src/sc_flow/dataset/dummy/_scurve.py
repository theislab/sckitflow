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

    def __init__(
        self,
        random_state: int = 42,
        n_samples: int = 1000,
        noise: float = 0.0,
    ):
        """Initialization"""
        super().__init__(random_state, n_samples, n_features=2, noise=noise)

    def _get_dataset_name(self) -> str:
        """Return identifier for this dataset type."""
        return "scurve"

    def _validate_additional_parameters(self, **kwargs):
        """Validate dataset-specific parameters."""
        noise = kwargs["noise"]

        if not isinstance(noise, int | float):
            raise TypeError(f"noise must be a number, got {type(noise)}")

        if noise < 0:
            raise ValueError(f"noise must be non-negative, got {noise}")

    def _generate(self, random_state, n_samples, n_features, noise) -> sc.AnnData:
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
        adata = self._create_anndata(X, y)

        # Step 3: Add scurve-specific metadata
        adata.uns["dataset_info"]["noise"] = noise

        return adata
