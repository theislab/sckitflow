from abc import ABC, abstractmethod
from typing import Any

import flax.linen as nn

from sc_flow.backends.jax._types import ArrayLike


class Solver(ABC, nn.Module):
    """Abstract base class for JAX solvers."""

    @abstractmethod
    def solve(
        self,
        source: ArrayLike | None = None,
        return_trajectory: bool = False,
        *,
        solver_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ArrayLike:
        """Solve method to be implemented by subclasses."""
        ...
