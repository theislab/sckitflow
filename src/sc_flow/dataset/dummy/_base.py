from abc import ABC, abstractmethod

import numpy as np
import scanpy as sc


class BaseDummyDataset(ABC):
    """
    Abstract base class for all dummy dataset generators.

    Attributes
    ----------
    random_state : int
        Random seed for reproducibility across all dataset types

    dataset_name : str
        Identifier for the dataset type (e.g., 'blobs', 'moons')
    """

    _dataset_name: str

    def __init__(self, random_state: int, n_samples: int, **kwargs):
        """
        Initialize the base dataset generator.

        Parameters
        ----------
        random_state : int, optional (default=42)
            Seed for random number generation to ensure reproducibility
        """
        self.random_state = random_state

        # Validate common parameters
        if not isinstance(n_samples, int):
            raise TypeError(f"n_samples must be an integer, got {type(n_samples)}")

        if n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {n_samples}")

        if not isinstance(random_state, int):
            raise TypeError(f"random_state must be an integer, got {type(random_state)}")

        self._validate_additional_parameters(**kwargs)
        self.adata = self._generate(random_state, n_samples, **kwargs)

    def _create_anndata(self, X: np.ndarray, y: np.ndarray | None = None, y_type: str | None = None) -> sc.AnnData:
        """
        Convert numpy arrays into AnnData format

        Parameters
        ----------
        X : np.ndarray, shape (n_observations, n_features)
            Data matrix where a row is an observation and a column is a feature

        y : np.ndarray, optional, shape (n_observations,)
            Cluster labels or class assignments for each observation

        Returns
        -------
        adata : sc.AnnData
            Annotated data object with standardized structure
        """
        # Step 1: Create the core AnnData structure
        adata = sc.AnnData(X=X)

        # Step 2: Add cluster labels if provided
        # Labels are stored as strings for categorical data
        if y is not None:
            if y_type == "category":
                adata.obs["Y"] = y.astype(str)
                adata.obs["Y"] = adata.obs["Y"].astype("category")

            else:
                adata.obs["Y"] = y.astype(float)

        # Step 3: Store metadata about the dataset
        adata.uns["dataset_info"] = {
            "name": self._dataset_name,
            "random_state": self.random_state,
        }

        return adata

    @abstractmethod
    def _validate_additional_parameters(self, **kwargs) -> None:
        """Validate dataset-specific parameters."""
        raise NotImplementedError

    @abstractmethod
    def _generate(self, **kwargs) -> sc.AnnData:
        """
        Generate the dataset with specific parameters.

        Parameters
        ----------
        **kwargs : dict
            Dataset-specific parameters

        Returns
        -------
        adata : sc.AnnData
            Generated dataset wrapped as AnnData
        """
        raise NotImplementedError
