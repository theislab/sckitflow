from typing import Any

import torch
from torch import Tensor

from sc_flow.backends.torch._types import TDevice
from sc_flow.backends.torch.solvers._solver import BaseSolver

__all__ = ["FMSolver"]


class FMSolver(BaseSolver):
    r"""Solver for Flow Map Matching (LMD) that iteratively applies the learned flow map.

    The dynamics must provide a function `map_fn(s, t, x)` that maps the state from time `s`
    to time `t`. The solver composes these maps over a discrete time grid.

    :param dynamics: The neural network module (e.g., MLPFlowMap) providing the flow map.
    :param method: Ignored; kept for compatibility with BaseSolver.
    :param device_id: Device to use for computations.
    :param vf_kwargs: Keyword arguments passed to `dynamics.get_vf_fn(...)`.
    """

    def __init__(
        self,
        dynamics,
        *,
        method: str | None = None,
        device_id: TDevice = "cpu",
        vf_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(dynamics=dynamics, method=method, device_id=device_id)
        self._vf_kwargs = vf_kwargs or {}
        self._map_fn = dynamics.get_vf_fn(**self._vf_kwargs)

    def solve(
        self,
        source: Tensor,
        time: Tensor,
        *,
        rtol: float = 1e-7,  # not used, kept for API compatibility
        atol: float = 1e-9,  # not used
        solver_kwargs: dict[str, Any] | None = None,
        return_trajectory: bool = False,
    ) -> Tensor:
        r"""Integrates the flow map over the given time grid.

        :param source: Initial state at `time[0]`.
        :param time: 1D tensor of time points (strictly increasing).
        :param return_trajectory: If True, returns all intermediate states; else only the final state.
        :returns: Tensor of states; shape = `(len(time), *source.shape)` if `return_trajectory=True`,
                  else shape = `source.shape`.
        """
        config = self._prepare_solve_config(source, time, solver_kwargs)
        t = config.time_on_device
        x = config.source_on_device

        # Ensure time is 1D and sorted
        if t.dim() > 1:
            t = t.squeeze()
        if not torch.all(t[1:] > t[:-1]):
            raise ValueError("Time points must be strictly increasing for FMSolver.")

        traj = [x]
        for i in range(len(t) - 1):
            s = t[i].expand(x.shape[0])  # shape: batch
            t_next = t[i + 1].expand(x.shape[0])
            x = self._map_fn(s, t_next, x)
            traj.append(x)

        if return_trajectory:
            return torch.stack(traj, dim=0)
        else:
            return x

    @property
    def map_fn(self):
        return self._map_fn
