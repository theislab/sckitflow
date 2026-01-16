# Import all dataset classes
from sc_flow.dataset.dummy._blobs import BlobsDataset
from sc_flow.dataset.dummy._checkerboard import CheckerboardDataset
from sc_flow.dataset.dummy._circles import CirclesDataset
from sc_flow.dataset.dummy._moons import MoonsDataset
from sc_flow.dataset.dummy._scurve import SCurveDataset
from sc_flow.dataset.dummy._swissroll import SwissRollDataset


class dummy:
    """
    Unified function to generate any dummy dataset.

    Parameters
    ----------
    dataset_type : str
        Name of the dataset to generate. Options:
        - 'blobs': Gaussian clusters
        - 'checkerboard': Grid pattern
        - 'circles': Concentric circles
        - 'moons': Interleaving half-circles
        - 's_curve': S-shaped 3D manifold
        - 'swiss_roll': Rolled 3D manifold

    **kwargs : additional parameters
        Dataset-specific parameters

    Returns
    -------
    adata : sc.AnnData
        Generated dataset wrapped in AnnData format

    Raises
    ------
    ValueError : If dataset_type is not recognized
    """

    def get(self, dataset_type: str, **kwargs):
        # Step 1: Define mapping from dataset names to classes
        dataset_registry = {
            "blobs": BlobsDataset,
            "checkerboard": CheckerboardDataset,
            "circles": CirclesDataset,
            "moons": MoonsDataset,
            "s_curve": SCurveDataset,
            "swiss_roll": SwissRollDataset,
        }

        # Step 2: Check if dataset type is valid
        if dataset_type not in dataset_registry:
            available = ", ".join(sorted(dataset_registry.keys()))
            raise ValueError(f"Unknown dataset type: '{dataset_type}'. Available options: {available}")

        # Step 3: Instantiate the appropriate dataset class
        dataset_class = dataset_registry[dataset_type]
        adata = dataset_class(**kwargs)

        return adata

    def list_datasets(self) -> dict:
        """
        List all available datasets with their descriptions.

        Returns
        -------
        datasets : dict
            Dictionary mapping dataset names to descriptions
        """
        return {
            "blobs": "Isotropic Gaussian blobs for clustering",
            "checkerboard": "Grid-like pattern of clusters",
            "circles": "Two concentric circles",
            "moons": "Two interleaving half-circles",
            "s_curve": "3D S-shaped manifold",
            "swiss_roll": "3D rolled manifold",
        }

    def get_dataset_info(self, dataset_type: str) -> dict:
        """
        Get detailed information about a specific dataset type.

        Parameters
        ----------
        dataset_type : str
            Name of the dataset

        Returns
        -------
        info : dict
            Dictionary with dataset details including parameters and use cases
        """
        dataset_info = {
            "blobs": {
                "n_samples": "Number of data points (default: 1000)",
                "n_features": "Number of dimensions (default: 2)",
                "centers": "Number of clusters (default: 3)",
                "cluster_std": "Cluster spread (default: 1.0)",
                "center_box": "Bounding box for cluster centers (default: (-10.0, 10.0))",
                "shuffle": "Whether to shuffle samples (default: True)",
                "random_state": "Seed for random number generator (default: 42)",
            },
            "checkerboard": {
                "n_samples": "Number of data points (default: 1000)",
                "n_features": "Number of dimensions (default: 2)",
                "noise": "Noise level (default: 0.0)",
                "minval": "Minimum value for grid (default: 10)",
                "maxval": "Maximum value for grid (default: 100)",
                "shuffle": "Whether to shuffle samples (default: True)",
                "random_state": "Seed for random number generator (default: 42)",
                "n_clusters": "Grid size as (rows, cols) (default: (3, 3))",
            },
            "circles": {
                "n_samples": "Number of data points (default: 1000)",
                "factor": "Inner to outer circle ratio (default: 0.8)",
                "noise": "Noise level (default: 0.0)",
                "random_state": "Seed for random number generator (default: 42)",
            },
            "moons": {
                "n_samples": "Number of data points (default: 1000)",
                "noise": "Noise level (default: 0.05)",
                "shuffle": "Whether to shuffle samples (default: True)",
                "random_state": "Seed for random number generator (default: 42)",
            },
            "s_curve": {
                "n_samples": "Number of data points (default: 1000)",
                "noise": "Noise level (default: 0.0)",
                "random_state": "Seed for random number generator (default: 42)",
            },
            "swiss_roll": {
                "n_samples": "Number of data points (default: 1000)",
                "noise": "Noise level (default: 0.0)",
                "hole": "Whether to include a hole in the Swiss roll (default: False)",
                "random_state": "Seed for random number generator (default: 42)",
            },
        }

        if dataset_type not in dataset_info:
            available = ", ".join(sorted(dataset_info.keys()))
            raise ValueError(f"Unknown dataset type: '{dataset_type}'. Available options: {available}")

        return dataset_info[dataset_type]
