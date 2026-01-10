from typing import Any, Literal

import diffrax as dfx
import jax
import jax.numpy as jnp
from diffrax import Euler, ODETerm

from sc_flow.backends.jax._types import ArrayLike, TVfFn
from sc_flow.backends.jax.nn import BaseVelocityField
from sc_flow.backends.jax.solvers.solver import Solver


class ODESolver(Solver):
    """Class for solving neural Ordinary Differential Equations (ODEs).

    :param method: The numerical integration scheme used by diffrax.
        When ``None`` it will be set to :class:`diffrax.Euler`.
    :type method: class: `diffrax.AbstractSolver | None`

    :param num_time_steps: Number of time steps used to construct the time grid over
        :math:`[0, 1]`. When :param:`dt0` is not explicitly passed via :param:`solver_kwargs`,
        the initial step size is set to ``1 / (num_time_steps - 1)``.
    :type num_time_steps: class: `int`

    :param device_id: Identifier for the JAX device on which the ODE is solved.
    :type device_id: class: `Literal["cpu", "gpu", "tpu"] | jax.Device`
    """

    def __init__(
        self,
        method: dfx.AbstractSolver | None = None,
        num_time_steps: int = 500,
        device_id: Literal["cpu", "gpu", "tpu"] = "cpu",
    ) -> None:
        self.method = method or Euler()
        self.num_time_steps = num_time_steps
        self.ts = jnp.linspace(0.0, 1.0, self.num_time_steps)

        if isinstance(device_id, str):
            candidates = [d for d in jax.devices() if d.platform == device_id]
            if not candidates:
                raise ValueError(f"No available device found for platform '{device_id}'")

            self.device = candidates[0]
        else:
            self.device = device_id

    def solve(
        self,
        vf: BaseVelocityField,
        source: ArrayLike,
        return_trajectory: bool = False,
        solver_kwargs: dict[str, Any] | None = None,
        **vf_kwargs: Any,
    ) -> ArrayLike:
        """Solve the ODE defined by a neural velocity field.

        This method constructs the vector field from :param:`vf` via
        :meth:`BaseVelocityField.get_vf_fn` and integrates the ODE over the interval
        :math:`[0, 1]` using :func:`diffrax.diffeqsolve`.

        :param vf: The neural velocity field defining the ODE dynamics. It must implement
            :meth:`BaseVelocityField.get_vf_fn`, which should return a callable of the form
            ``vf_fn(t, x, args) -> dx_dt`` compatible with diffrax.
        :type vf: class: `BaseVelocityField`

        :param source: The initial state :math:`x_{t=0}` from which the ODE is integrated.
        :type source: class: `ArrayLike`

        :param return_trajectory: When set to ``True``, the full solution trajectory is
            returned as an array of shape ``(num_time_steps, *source.shape)``. When ``False``, only the final state at
            :math:`t = 1` is returned.
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

        :param vf_kwargs: Additional keyword arguments forwarded to
            :meth:`BaseVelocityField.get_vf_fn`. These can be used to configure the
            behavior of the velocity field at solve time (e.g. conditioning, extra
            parameters, etc.).
        :type vf_kwargs: class: `dict[str, Any]`

        :returns: Either the full solution trajectory or the final state at
            :math:`t = 1`, depending on :param:`return_trajectory`.
        :rtype: class: `ArrayLike`
        """
        if solver_kwargs is None:
            solver_kwargs = {}

        dt0 = 1.0 / (self.num_time_steps - 1)
        dt0 = solver_kwargs.pop("dt0", dt0)
        max_steps = solver_kwargs.pop("max_steps", 10_000)

        stepsize_controller = solver_kwargs.pop("stepsize_controller", None)
        if stepsize_controller is None:
            stepsize_controller = dfx.ConstantStepSize()
        elif not isinstance(stepsize_controller, dfx.AbstractStepSizeController):
            raise TypeError("options['stepsize_controller'] must be an instance of diffrax.AbstractStepSizeController.")

        vector_field: TVfFn = vf.get_vf_fn(**vf_kwargs)
        terms = ODETerm(vector_field)

        source = jax.device_put(source, self.device)

        if return_trajectory:
            saveat = dfx.SaveAt(ts=self.ts)
        else:
            saveat = dfx.SaveAt(t1=True)

        trajectory = dfx.diffeqsolve(
            terms,
            solver=self.method,
            t0=0.0,
            t1=1.0,
            dt0=dt0,
            y0=source,
            saveat=saveat,
            max_steps=max_steps,
            stepsize_controller=stepsize_controller,
            **solver_kwargs,
        )

        if return_trajectory:
            return trajectory.ys
        else:
            return trajectory.ys[-1]
