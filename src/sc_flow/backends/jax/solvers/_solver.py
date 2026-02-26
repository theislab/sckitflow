from abc import ABC, abstractmethod
from typing import Any, Generic

import diffrax as dfx
import flax.linen as nn
import jax
import jax.numpy as jnp

from sc_flow.backends.jax._types import ArrayLike, SolverConfig, TDevice, TSolverDynamics
from sc_flow.backends.jax._utils import get_jax_device


class BaseSolver(Generic[TSolverDynamics], ABC, nn.Module):
    """Abstract base class for JAX solvers."""

    def __init__(self, dynamics: TSolverDynamics, method: str | dfx.AbstractSolver | None, device_id=TDevice) -> None:
        super().__init__()
        self._dynamics = dynamics
        self._method = method or dfx.Euler()
        self._device = get_jax_device(device_id)

    @abstractmethod
    def solve(
        self,
        source: ArrayLike | None = None,
        t0: float = 0.0,
        t1: float = 1.0,
        *,
        num_time_steps: int = 500,
        return_trajectory: bool = False,
        solver_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ArrayLike:
        """Solve method to be implemented by subclasses."""
        ...

    def _prepare_solve_config(
        self,
        source: ArrayLike,
        t0: float,
        t1: float,
        num_time_steps: int,
        return_trajectory: bool,
        solver_kwargs: dict[str, Any] | None = None,
    ) -> SolverConfig:
        """Prepare common solver configuration.

        Extracts and processes common parameters from solver_kwargs and prepares
        the source data on the appropriate device.

        :param source: Initial state for the solver.
        :param num_time_steps: Number of time steps for the integration.
        :param return_trajectory: Whether to save the full trajectory.
        :param solver_kwargs: Additional solver configuration.
        :returns: SolverConfig with all prepared parameters.
        """
        solver_kwargs = solver_kwargs or {}
        ts = jnp.linspace(t0, t1, num_time_steps)
        dt0 = solver_kwargs.pop("dt0", 1.0 / (num_time_steps - 1))
        max_steps = solver_kwargs.pop("max_steps", 10_000)

        stepsize_controller = solver_kwargs.pop("stepsize_controller", None)
        if stepsize_controller is None:
            stepsize_controller = dfx.ConstantStepSize()
        elif not isinstance(stepsize_controller, dfx.AbstractStepSizeController):
            msg = "solver_kwargs['stepsize_controller'] must be an instance of diffrax.AbstractStepSizeController."
            raise TypeError(msg)
        elif isinstance(stepsize_controller, dfx.StepTo):
            dt0 = None

        if return_trajectory:
            saveat = dfx.SaveAt(ts=ts)
        else:
            saveat = dfx.SaveAt(t1=True)
        source_on_device = jax.device_put(source, self._device)

        return SolverConfig(
            dt0=dt0,
            max_steps=max_steps,
            stepsize_controller=stepsize_controller,
            saveat=saveat,
            source_on_device=source_on_device,
            remaining_kwargs=solver_kwargs,
        )

    @property
    def dynamics(self) -> TSolverDynamics:
        """Get the dynamics associated with the solver."""
        return self._dynamics

    @property
    def device_id(self) -> TDevice:
        """Get the device identifier for the solver."""
        return self._device

    @property
    def method(self) -> str | dfx.AbstractSolver:
        """Get the integration method used by the solver."""
        return self._method
