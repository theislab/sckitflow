from typing import Any, Literal

import diffrax as dfx
import jax
import jax.numpy as jnp
from diffrax import ControlTerm, Euler, ODETerm
from jax.lib import xla_client

from sc_flow.backends.jax._types import ArrayLike, TVfFn
from sc_flow.backends.jax.solvers.solver import Solver


class SDESolver(Solver):
    """Base Class for SDE Solvers"""

    def __init__(
        self,
        method: dfx.AbstractSolver | None = None,
        num_time_steps: int = 500,
        device_id: Literal["cpu", "gpu", "tpu"] | xla_client.Device = "cpu",
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
        """Solve the SDE defined by the drift and diffusion terms."""
        if solver_kwargs is None:
            solver_kwargs = {}

        dt0 = 1.0 / (self.num_time_steps - 1)
        dt0 = solver_kwargs.pop("dt0", dt0)
        max_steps = solver_kwargs.pop("max_steps", 10_000)

        stepsize_controller = solver_kwargs.pop("stepsize_controller", None)
        if stepsize_controller is not None:
            if not isinstance(stepsize_controller, dfx.AbstractStepSizeController):
                raise TypeError(
                    "options['stepsize_controller'] must be an instance of diffrax.AbstractStepSizeController."
                )

        if brownian_motion is None:
            brownian_motion = dfx.UnsafeBrownianPath(
                t0=0.0,
                t1=1.0,
                shape=source.shape,
                key=jax.random.PRNGKey(0),
            )

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

        return trajectory.ys
