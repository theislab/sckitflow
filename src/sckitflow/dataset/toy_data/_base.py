from abc import ABC, abstractmethod

import numpy as np
from anndata import AnnData


class BaseDummyDataset(ABC):
    """
    Abstract base class for dummy dataset generators.

    Parameters
    ----------
    random_state : int
        Random seed for reproducibility across all dataset types

    n_samples : int
        Number of samples to generate

    **kwargs
        Dataset-specific parameters forwarded to `_generate`

    Attributes
    ----------
    adata : AnnData
        Generated dataset in AnnData format

    dataset_name : str
        Name of the dataset type

    Note
    ----
    Subclasses must:
    - Implement `_generate` method
    - Define `_dataset_name` attribute
    """

    _dataset_name: str

    def __init__(self, random_state: int, n_samples: int, **kwargs):
        """Initialize the base dataset generator."""
        self.random_state = random_state
        self.adata = self._generate(random_state, n_samples, **kwargs)

    def _create_anndata(self, X: np.ndarray, y: np.ndarray | None = None, y_type: str | None = None) -> AnnData:
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
        adata : AnnData
            Annotated data object with standardized structure
        """
        # Step 1: Create the core AnnData structure
        # float32 is the dtype the torch models run in; sklearn hands us float64. Extraction never casts,
        # so a float64 `X` would only surface much later as a `Double` vs `Float` matmul error.
        adata = AnnData(X=X.astype(np.float32))

        # Step 2: Add cluster labels if provided
        # Labels are stored as strings for categorical data
        if y is not None:
            if y_type == "category":
                adata.obsm["Y"] = y.astype(str)
                # adata.obsm["Y"] = adata.obsm["Y"].astype("category")

            else:
                adata.obsm["Y"] = y.astype(np.float32)

        # Step 3: Store metadata about the dataset
        adata.uns["dataset_info"] = {
            "name": self._dataset_name,
            "random_state": self.random_state,
        }

        return adata

    @property
    def dataset_name(self) -> str:
        """Get the dataset name."""
        return self._dataset_name

    @abstractmethod
    def _generate(self, **kwargs) -> AnnData:
        """
        Generate the dataset with specific parameters.

        Parameters
        ----------
        **kwargs : dict
            Dataset-specific parameters

        Returns
        -------
        adata : AnnData
            Generated dataset wrapped as AnnData
        """
        raise NotImplementedError
