from abc import ABC, abstractmethod

from torch import Tensor, nn


class Solver(ABC, nn.Module):
    "Abstract Base Class for Solvers"
    
    @abstractmethod
    def solve(self, source: Tensor = None) -> Tensor:
        ...