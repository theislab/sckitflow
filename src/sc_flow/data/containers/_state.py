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

    def _slice_with_array(
        self,
        idxs: np.ndarray,
    ) -> "StateData":
        X = self.X[idxs]
        return self.__class__(X)

    @property
    def n_obs(self) -> int:
        return self.X.shape[0]
