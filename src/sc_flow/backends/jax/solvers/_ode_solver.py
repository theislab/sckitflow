from typing import Any

import diffrax as dfx
from diffrax import ODETerm

from sc_flow.backends.jax._types import ArrayLike, TDevice, TODEDynamics, TVfFn
from sc_flow.backends.jax._utils import get_ode_solver
from sc_flow.backends.jax.solvers._solver import BaseSolver


class ODESolver(BaseSolver[TODEDynamics]):
    r"""Class for solving neural Ordinary Differential Equations (ODEs).

    :param dynamics: Velocity field providing the time-dependent dynamics. Must implement :meth:`BaseVelocityField.get_vf_fn`.
    :type dynamics: class:`TODEDynamics`

    :param method: The numerical integration scheme used by diffrax.
        When ``None`` it will be set to :class:`diffrax.Euler`.
    :type method: class: `diffrax.AbstractSolver | None`

    :param num_time_steps: Number of time steps used to construct the time grid over
        :math:`[0, 1]`. When :param:`dt0` is not explicitly passed via :param:`solver_kwargs`,
        the initial step size is set to ``1 / (num_time_steps - 1)``.
    :type num_time_steps: class: `int`

    :param device_id: Identifier for the JAX device on which the ODE is solved.
    :type device_id: class: `TDevice`

    :param vf_kwargs: Additional keyword arguments forwarded to
    :meth:`BaseVelocityField.get_vf_fn`. These can be used to configure the
    behavior of the velocity field at solve time (e.g. conditioning, extra
    parameters, etc.).
    :type vf_kwargs: class: `dict[str, Any]`
    """

    def __init__(
        self,
        dynamics: TODEDynamics,
        *,
        method: dfx.AbstractSolver | None = None,
        device_id: TDevice = "cpu",
        vf_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(dynamics=dynamics, method=method, device_id=device_id)

        self._method = get_ode_solver(method)
        vf_kwargs = vf_kwargs or {}
        self._vf = dynamics.get_vf_fn(**vf_kwargs)

    def solve(
        self,
        source: ArrayLike,
        t0: float = 0.0,
        t1: float = 1.0,
        *,
        num_time_steps: int = 500,
        return_trajectory: bool = False,
        solver_kwargs: dict[str, Any] | None = None,
    ) -> ArrayLike:
        r"""Solve the ODE defined by a neural velocity field.

        :param source: The initial state :math:`x_{t=0}` from which the ODE is integrated.
        :type source: class: `ArrayLike`

        :param t0: Initial time for the ODE integration. Defaults to ``0.0``.
        :type t0: class: `float`

        :param t1: Final time for the ODE integration. Defaults to ``1.0``.
        :type t1: class: `float`

        :param return_trajectory: When set to ``True``, the full solution trajectory is
            returned as an array of shape ``(num_time_steps, *source.shape)``. When ``False``, only the final state at
            :math:`t = t1` is returned.
        :type return_trajectory: class: `bool`

        :param solver_kwargs: (Optional) Dictionary of keyword arguments forwarded to
            :func:`diffrax.diffeqsolve`. The following keys are handled explicitly and
            removed from this dictionary:

            * ``"dt0"``: Initial time step size. When not provided, it is set to
              ``1 / (num_time_steps - 1)``.
            * ``"max_steps"``: Maximum number of internal solver steps. Defaults to
              ``10_000``.
            * ``"stepsize_controller"``: Optional instance of
              :class:`diffrax.AbstractStepSizeController`. When not provided, a
              :class:`diffrax.ConstantStepSize` controller is used.

            Any additional entries in :param:`solver_kwargs` are passed to
            :func:`diffrax.diffeqsolve`.
        :type solver_kwargs: class: `dict[str, Any] | None`

        :returns: Either the full solution trajectory or the final state at
            :math:`t = 1`, depending on :param:`return_trajectory`.
        :rtype: class: `ArrayLike`
        """
        config = self._prepare_solve_config(
            source=source,
            t0=t0,
            t1=t1,
            num_time_steps=num_time_steps,
            return_trajectory=return_trajectory,
            solver_kwargs=solver_kwargs,
        )
        terms = ODETerm(self._vf)
        args = config.remaining_kwargs.pop("args", None)

        trajectory = dfx.diffeqsolve(
            terms,
            solver=self._method,
            t0=t0,
            t1=t1,
            dt0=config.dt0,
            y0=config.source_on_device,
            saveat=config.saveat,
            max_steps=config.max_steps,
            stepsize_controller=config.stepsize_controller,
            args=args,
            **config.remaining_kwargs,
        )

        if return_trajectory:
            return trajectory.ys
        else:
            return trajectory.ys[-1]

    @property
    def vf(self) -> TVfFn:
        """Get the velocity field function used by the ODE solver."""
        return self._vf
