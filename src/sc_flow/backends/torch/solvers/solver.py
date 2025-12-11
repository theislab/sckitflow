from abc import ABC, abstractmethod
from typing import Any

from torch import Tensor, nn


class Solver(ABC, nn.Module):
    """Abstract Base Class for Solvers"""

    @abstractmethod
    def solve(self, source: Tensor | None = None, **kwargs: Any) -> Tensor:
        """Solve method to be implemented by subclasses."""
        ...
