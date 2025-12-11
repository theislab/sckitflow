from typing import Any, Literal

import diffrax as dfx
import jax
import jax.numpy as jnp
from diffrax import Euler
from jax.lib import xla_client

from sc_flow.backends.jax._types import ArrayLike, TVfFn
from sc_flow.backends.jax.nn import BaseVelocityField
from sc_flow.backends.jax.solvers.solver import Solver


class ODESolver(Solver):
    """Base Class for ODE Solvers"""

    def __init__(
        self,
        vf: BaseVelocityField,
        method: dfx.AbstractSolver | None = None,
        num_time_steps: int = 500,
        atol: float = 1e-5,
        rtol: float = 1e-5,
        device_id: Literal["cpu", "gpu", "tpu"] | xla_client.Device = "cpu",
    ) -> None:
        self.vf = vf
        self.method = method or Euler()
        self.num_time_steps = num_time_steps
        self.atol = atol
        self.rtol = rtol
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
        return_trajectory: bool = False,
        options: dict[str, Any] | None = None,
        **vf_kwargs: Any,
    ) -> ArrayLike:
        """Solve the ODE defined by the velocity field."""
        if options is None:
            options = {}

        dt0 = 1.0 / (self.num_time_steps - 1)
        dt0 = options.pop("dt0", dt0)
        max_steps = options.pop("max_steps", 10_000)

        stepsize_controller = options.pop("stepsize_controller", None)
        if stepsize_controller is not None:
            if not isinstance(stepsize_controller, dfx.AbstractStepSizeController):
                raise TypeError(
                    "options['stepsize_controller'] must be an instance of diffrax.AbstractStepSizeController."
                )

        terms: TVfFn = self.vf.get_vf_fn(**vf_kwargs)
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
            rtol=self.rtol,
            atol=self.atol,
            max_steps=max_steps,
            stepsize_controller=stepsize_controller,
            **options,
        )

        return trajectory.ys
