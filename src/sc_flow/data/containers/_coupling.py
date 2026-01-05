from dataclasses import dataclass

import numpy as np

from sc_flow.data.containers._base import BaseData
from sc_flow.data.containers._state import StateData

__all__ = ["CouplingData"]


@dataclass(frozen=True)
class CouplingData(BaseData):
    """Container class for coupling data.

    :param state_lin: Container for the linear term of samples.
    :type state_lin: class: `StateData`

    :param state_quad: Container for the quadratic term of samples.
        Defaults to `None`.
    :type state_quad: class: `StateData | None`
    """

    state_lin: StateData
    state_quad: StateData | None = None

    def __post_init__(self):
        if self.state_quad is not None:
            self.state_lin._assert_same_n_obs(self.state_quad)

    def __repr__(self) -> str:
        parts = [f"n_obs={self.n_obs}"]

        lin_shape = self.state_lin.X.shape
        lin_dims = lin_shape[1:] if len(lin_shape) > 1 else ()
        parts.append(f"linear(spatial_dims={lin_dims})")

        if self.state_quad is not None:
            quad_shape = self.state_quad.X.shape
            quad_dims = quad_shape[1:] if len(quad_shape) > 1 else ()
            parts.append(f"quadratic(spatial_dims={quad_dims})")
        else:
            parts.append("quadratic=None")

        return f"{self.__class__.__name__}({', '.join(parts)})"

    def _slice(self, idxs: np.ndarray | slice) -> "CouplingData":
        state_lin = self.state_lin.slice_with_array(idxs)
        state_quad = None if self.state_quad is None else self.state_quad.slice_with_array(idxs)
        return self.__class__(
            state_lin,
            state_quad=state_quad,
        )

    def assert_same_spatial_dims(
        self,
        other: "CouplingData",
    ) -> None:
        """Checks that the current coupling data shares the same number of spatial dimensions with another.

        The check is done over the linear term only, as this will be the factor in the product space that
        will be shared by both source and target distributions. The remaining quadratic terms do not need
        to align.
        """
        n_dims_self = self.state_lin.X.shape[1]
        n_dims_other = other.state_lin.X.shape[1]
        if n_dims_self != n_dims_other:
            msg = (
                "Coupling data should share the same number of spatial "
                f"dimensions for the linear term, found {n_dims_self} and {n_dims_other}."
            )
            raise ValueError(msg)

    @classmethod
    def init_from_state_data(
        cls,
        state_data: StateData,
        n_shared_dims: int | None = None,
    ) -> "CouplingData":
        """Initializes the coupling data from base state data and number of shared dimensions.

        :param state_data: The underlying state data.
        :type state_data: class: `StateData`

        :param n_shared_dims: The number of shared dimensions. Defaults to `None` (all dimensions).
        :type n_shared_dims: class: `int | None`
        """
        n_dims = state_data.X.shape[1]
        if n_shared_dims is not None:
            if n_shared_dims >= n_dims:
                msg = (
                    "The number of shared spatial dimensions should "
                    "be strictly smaller the the number of available dimenions. "
                    f"Queried {n_shared_dims} on state data of shape {state_data.X.shape}"
                )
                raise ValueError(msg)
            state_lin = StateData(state_data.X[:, :n_shared_dims])
            state_quad = StateData(state_data.X[:, n_shared_dims:])
        else:
            state_lin = state_data
            state_quad = None
        return cls(state_lin, state_quad)

    @property
    def n_obs(self) -> int:
        return self.state_lin.n_obs
