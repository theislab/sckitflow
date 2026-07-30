from typing import Any

from torch import Tensor
from torchdiffeq import odeint

from sckitflow.backends.torch._types import TDevice, TODEDynamics, TVfFn
from sckitflow.backends.torch.solvers._solver import BaseSolver


class ODESolver(BaseSolver[TODEDynamics]):
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

        vf_kwargs = vf_kwargs or {}
        self._vf = dynamics.get_vf_fn(**vf_kwargs)

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
        config = self._prepare_solve_config(source, time, solver_kwargs)

        trajectory = odeint(
            self._vf,
            config.source_on_device,
            config.time_on_device,
            rtol=rtol,
            atol=atol,
            method=self._method,
            **config.remaining_kwargs,
        )

        if return_trajectory:
            return trajectory
        else:
            return trajectory[-1]

    @property
    def vf(self) -> TVfFn:
        """Get the velocity field associated with the ODE solver."""
        return self._vf
