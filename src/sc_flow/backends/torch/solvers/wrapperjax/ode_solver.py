from typing import Any

import numpy as np
import torch
import torchax
from torch import Tensor
from torchax.interop import jax_view, torch_view

from sc_flow.backends.jax.solvers import ODESolver as JAXODESolver
from sc_flow.backends.torch._types import TDevice, TODEDynamics
from sc_flow.backends.torch.solvers.solver import BaseSolver

from ._utils import dynamics_wrapper, map_torch_method_to_jax


class WrappedODESolver(BaseSolver[TODEDynamics]):
    r"""Class for solving deterministic ordinary differential equations (ODEs) with :func:`torchdiffeq.odeint`.

    :param dynamics: Velocity field providing the time-dependent dynamics. Must implement :meth:`BaseVelocityField.get_vf_fn`.
    :type dynamics: class:`TODEDynamics`

    :param method: (Optional) Integration scheme used by ``torchdiffeq``. Defaults to ``"euler"``. Other valid options depend on ``torchdiffeq``.
    :type method: class:`str`

    :param device_id: (Optional) Identifier for the target compute device.
    :type device_id: class:`TDevice`

    :param vf_kwargs: (Optional) Keyword arguments passed to
            :meth:`BaseVelocityField.get_vf_fn`.
        :type vf_kwargs: class:`dict[str, Any] | None`
    """

    def __init__(
        self,
        dynamics: TODEDynamics,
        *,
        method: str = "euler",
        device_id: TDevice = "cpu",
        vf_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(dynamics=dynamics, method=method, device_id=device_id)
        torchax.enable_globally()
        vf_kwargs = vf_kwargs or {}
        self._dynamics = dynamics
        self._vf_kwargs = vf_kwargs
        self._jax_dynamics = dynamics_wrapper(dynamics)

        jax_method = map_torch_method_to_jax(method)

        self._jax_solver = JAXODESolver(
            dynamics=self._jax_dynamics,
            method=jax_method,
            device_id=device_id,
            vf_kwargs=vf_kwargs,
        )

    def solve(
        self,
        source: Tensor,
        time: Tensor,
        *,
        rtol: float = 1e-7,
        atol: float = 1e-9,
        solver_kwargs: dict[str, Any] | None = None,
        return_trajectory: bool = False,
    ) -> Tensor:
        r"""Integrates the ODE defined by the provided velocity field.

        :param source: Initial state of the ODE.
        :type source: class:`torch.Tensor`

        :param time: Time grid over which to integrate the ODE.
        :type time: class:`torch.Tensor`

        :param rtol: Relative tolerance for the ODE solver.
        :type rtol: class:`float`

        :param atol: Absolute tolerance for the ODE solver.
        :type atol: class:`float`

        :param solver_kwargs: (Optional) Keyword arguments forwarded directly to
            :func:`torchdiffeq.odeint`.
        :type solver_kwargs: class:`dict[str, Any] | None`

        :param return_trajectory: When ``True``, returns the full trajectory, else returns only the final state at the last time point.
        :type return_trajectory: class:`bool`

        :returns: Either the full state trajectory or the final state depending on
            :param:`return_trajectory`.
        :rtype: class:`torch.Tensor`
        """
        source_j = source if source.device.type == "jax" else source.to("jax")
        time_j = time if time.device.type == "jax" else time.to("jax")

        source_jax = jax_view(source_j)
        t0 = float(time_j[0].item())
        t1 = float(time_j[-1].item())
        num_time_steps = int(time_j.shape[0])

        trajectory = self._jax_solver.solve(
            source_jax,
            t0,
            t1,
            num_time_steps=num_time_steps,
            return_trajectory=return_trajectory,
            solver_kwargs=solver_kwargs,
        )

        trajectory = torch_view(trajectory)
        trajectory = torch.from_numpy(np.array(trajectory)).to(self._device)

        if return_trajectory:
            return trajectory
        else:
            return trajectory[-1]

    @property
    def vf_kwargs(self) -> dict[str, Any]:
        """Get the velocity field associated with the ODE solver."""
        return self._vf_kwargs

    @property
    def dynamics(self) -> TODEDynamics:
        """Get the dynamics object associated with the ODE solver."""
        return self._dynamics
