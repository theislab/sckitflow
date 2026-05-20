from collections.abc import Collection
from dataclasses import dataclass

import numpy as np

from sc_flow.data.containers._base import BaseData

__all__ = ["StateData"]


@dataclass(frozen=True)
class StateData(BaseData):
    """Container class for state data.

    :param X: Array containing the underlying data.
    :type X: class: `np.ndarray`
    """

    X: np.ndarray

    def __repr__(self) -> str:
        shape = self.X.shape
        n_obs = shape[0]
        spatial_dims = shape[1:] if len(shape) > 1 else ()
        return f"{self.__class__.__name__}(n_obs={n_obs}, spatial_dims={spatial_dims})"

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(
        self,
        idxs: np.ndarray | slice,
    ) -> "StateData":
        X = self.X[idxs]
        return self.__class__(X)

    @classmethod
    def concat_collection(
        cls,
        collection: "Collection[StateData]",
    ) -> "StateData":
        """Concatenates a collection of instances into a single object."""
        # store for data
        X_collection = []

        # define base dimension
        ref_dim = -1

        # iterate over each element
        for idx, element in enumerate(collection):
            # get reference dimension from first element
            if idx == 0:
                ref_dim = element.X.shape[1]

            # get data for current element
            X_element = element.X

            # raise error if dimensions are mismatching
            if ref_dim != X_element.shape[1]:
                raise ValueError("Mismatching dimensions for concatenated object")

            X_collection.append(X_element)
        X = np.concatenate(X_collection)
        return cls(X)
