from typing import Any, Literal

import diffrax as dfx
import jax
import jax.numpy as jnp
from diffrax import ConstantStepSize, ControlTerm, Euler, ODETerm

from sc_flow.backends.jax._types import ArrayLike, TVfFn
from sc_flow.backends.jax.solvers._utils import validate_sde_solve_params
from sc_flow.backends.jax.solvers.solver import Solver


class SDESolver(Solver):
    r"""Class for solving neural Stochastic Differential Equations (SDEs).

    :param method: (Optional) The numerical integration scheme used for the SDE.
        When ``None`` it defaults to :class:`diffrax.Euler`.
    :type method: class: `diffrax.AbstractSolver | None`

    :param num_time_steps: Number of time steps used to construct the time grid over
        :math:`[0, 1]`. When :param:`dt0` is not explicitly passed via :param:`solver_kwargs`,
        the initial step size is set to ``1 / (num_time_steps - 1)``.
    :type num_time_steps: class: `int`

    :param device_id: Identifier for the JAX device on which the SDE is solved.
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
        source: ArrayLike,
        drift_fn: TVfFn,
        diffusion_fn: TVfFn,
        brownian_motion: dfx.AbstractBrownianPath | None = None,
        *,
        return_trajectory: bool = False,
        solver_kwargs: dict[str, Any] | None = None,
    ) -> ArrayLike:
        r"""Solve the SDE defined by drift and diffusion vector fields.

        :param source: Initial state :math:`x_{t=0}` from which the SDE is integrated.
        :type source: class: `ArrayLike`

        :param drift_fn: Drift vector field implementing
            ``drift_fn(t, x, args) -> drift``. It should be compatible with diffrax’s
            :class:`diffrax.ODETerm` interface.
        :type drift_fn: class: `TVfFn`

        :param diffusion_fn: Diffusion vector field implementing
            ``diffusion_fn(t, x, args) -> diffusion``. It is wrapped by
            :class:`diffrax.ControlTerm` together with the Brownian path.
        :type diffusion_fn: class: `TVfFn`

        :param brownian_motion: (Optional) Brownian path driving the diffusion term.
            When ``None``, a :class:`diffrax.VirtualBrownianTree` is constructed over
            :math:`[0, 1]` with shape ``source.shape`` and a default tolerance of ``1e-3``.
        :type brownian_motion: class: `diffrax.AbstractBrownianPath | None`

        :param return_trajectory: When ``True``, the full solution trajectory is returned
            otherwise only the final state at :math:`t = 1`
            is returned.
        :type return_trajectory: class: `bool`

        :param solver_kwargs: (Optional) Dictionary of keyword arguments forwarded to
            :func:`diffrax.diffeqsolve`. The following keys are handled explicitly and
            removed from this dictionary:

            * ``"dt0"``: Initial time step size. When not provided, it is set to
              ``1 / (num_time_steps - 1)``.
            * ``"max_steps"``: Maximum number of internal solver steps. Defaults to
              ``10_000``.
            * ``"stepsize_controller"``: Stepsize controller used by the solver.
              Defaults to :class:`diffrax.ConstantStepSize`. This must be compatible
              with the chosen Brownian path and adjoint settings; otherwise
              :func:`validate_sde_solve_params` will raise an error.
            * ``"adjoint"``: (Optional) Adjoint method configuration passed on to
              :func:`diffrax.diffeqsolve` (e.g. for backpropagation through the SDE).

            Any remaining entries in :param:`solver_kwargs` are passed to
            :func:`diffrax.diffeqsolve`.
        :type solver_kwargs: class: `dict[str, Any] | None`

        :returns: Either the full solution trajectory over :attr:`ts` or the final state at
            :math:`t = 1`, depending on :param:`return_trajectory`.
        :rtype: class: `ArrayLike`
        """
        if solver_kwargs is None:
            solver_kwargs = {}

        dt0 = 1.0 / (self.num_time_steps - 1)
        dt0 = solver_kwargs.pop("dt0", dt0)
        max_steps = solver_kwargs.pop("max_steps", 10_000)

        stepsize_controller = solver_kwargs.pop("stepsize_controller", ConstantStepSize())
        adjoint = solver_kwargs.pop("adjoint", None)

        if brownian_motion is None:
            brownian_motion = dfx.VirtualBrownianTree(
                t0=0.0,
                t1=1.0,
                shape=source.shape,
                tol=1e-3,
                key=jax.random.PRNGKey(0),
            )

        validate_sde_solve_params(stepsize_controller, brownian_motion, adjoint)

        if adjoint is not None:
            solver_kwargs["adjoint"] = adjoint

        terms = dfx.MultiTerm(
            ODETerm(drift_fn),
            ControlTerm(diffusion_fn, brownian_motion),
        )
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
