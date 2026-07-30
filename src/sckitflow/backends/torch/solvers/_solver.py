from abc import ABC, abstractmethod
from typing import Any, Generic

from torch import Tensor, nn

from sckitflow.backends.torch._types import SolverConfig, TDevice, TSolverDynamics
from sckitflow.backends.torch._utils import get_torch_device


class BaseSolver(Generic[TSolverDynamics], ABC, nn.Module):
    """Abstract Base Class for Solvers"""

    def __init__(self, dynamics: TSolverDynamics, *, method: str | None, device_id: TDevice = "cpu") -> None:
        super().__init__()
        self._dynamics = dynamics
        self._device = get_torch_device(device_id)
        self._method = method or "euler"

    def _prepare_solve_config(
        self,
        source: Tensor,
        time: Tensor,
        solver_kwargs: dict[str, Any] | None = None,
    ) -> SolverConfig:
        """Prepare common solver configuration.

        Processes solver_kwargs and moves tensors to the appropriate device.

        :param source: Initial state for the solver.
        :param time: Time grid for integration.
        :param solver_kwargs: Additional solver configuration.
        :returns: SolverConfig with all prepared parameters.
        """
        solver_kwargs = solver_kwargs or {}
        source_on_device = source.to(self._device)
        time_on_device = time.to(self._device)

        return SolverConfig(
            source_on_device=source_on_device,
            time_on_device=time_on_device,
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
    def method(self) -> str:
        """Get the integration method used by the solver."""
        return self._method

    @abstractmethod
    def solve(self, *, source: Tensor | None = None, **kwargs: Any) -> Tensor:
        """Solve method to be implemented by subclasses."""
        ...
