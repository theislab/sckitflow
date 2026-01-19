from typing import Any, Literal

from torch import Tensor, device
from torchdiffeq import odeint

from sc_flow.backends.torch.nn._vf import BaseVelocityField
from sc_flow.backends.torch.solvers.solver import Solver


class ODESolver(Solver):
    r"""Class for solving deterministic ordinary differential equations (ODEs) with :func:`torchdiffeq.odeint`.

    :param method: (Optional) Integration scheme used by ``torchdiffeq``. Defaults to ``"euler"``. Other valid options depend on ``torchdiffeq``.
    :type method: class:`str`

    :param device_id: (Optional) Identifier for the target compute device. Choices are ``"cpu"`` or ``"cuda"``. Defaults to ``"cpu"``.
    :type device_id: class:`Literal["cuda", "cpu"]`
    """

    def __init__(
        self,
        method: str = "euler",
        device_id: Literal["cuda", "cpu"] = "cpu",
    ) -> None:
        super().__init__()
        self.method = method
        self.device = device(device_id)

    def solve(
        self,
        vf: BaseVelocityField,
        source: Tensor,
        time: Tensor,
        *,
        rtol: float,
        atol: float,
        vf_kwargs: dict[str, Any] | None = None,
        solver_kwargs: dict[str, Any] | None = None,
        return_trajectory: bool = True,
    ) -> Tensor:
        r"""Integrates the ODE defined by the provided velocity field.

        :param vf: Velocity field providing the time-dependent dynamics. Must implement :meth:`BaseVelocityField.get_vf_fn`.
        :type vf: class:`BaseVelocityField`

        :param source: Initial state of the ODE.
        :type source: class:`torch.Tensor`

        :param time: Time grid over which to integrate the ODE.
        :type time: class:`torch.Tensor`

        :param rtol: Relative tolerance for the ODE solver.
        :type rtol: class:`float`

        :param atol: Absolute tolerance for the ODE solver.
        :type atol: class:`float`

        :param vf_kwargs: (Optional) Keyword arguments passed to
            :meth:`BaseVelocityField.get_vf_fn`.
        :type vf_kwargs: class:`dict[str, Any] | None`

        :param solver_kwargs: (Optional) Keyword arguments forwarded directly to
            :func:`torchdiffeq.odeint`.
        :type solver_kwargs: class:`dict[str, Any] | None`

        :param return_trajectory: When ``True``, returns the full trajectory, else returns only the final state at the last time point.
        :type return_trajectory: class:`bool`

        :returns: Either the full state trajectory or the final state depending on
            :param:`return_trajectory`.
        :rtype: class:`torch.Tensor`
        """
        if solver_kwargs is None:
            solver_kwargs = {}

        if vf_kwargs is None:
            vf_kwargs = {}

        diff_eqn = vf.get_vf_fn(**vf_kwargs)

        source = source.to(self.device)
        time = time.to(self.device)

        trajectory = odeint(
            diff_eqn,
            source,
            time,
            rtol=rtol,
            atol=atol,
            method=self.method,
            **solver_kwargs,
        )

        if return_trajectory:
            return trajectory
        else:
            return trajectory[-1]
