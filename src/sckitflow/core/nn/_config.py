from abc import ABC, abstractmethod

import torch
from scfit.registry import Component

from sckitflow.data._dims import DataDimensions

__all__ = ["ModuleConfig"]


class ModuleConfig(Component, ABC):
    """Family base for anything that configures a neural module from data dimensionality.

    Subclasses must be decorated with ``@component(...)`` to be
    instantiable (``Component``'s family-base check enforces this) and
    must override :meth:`build`.

    The return type is deliberately loose (``torch.nn.Module``) so each
    subclass can narrow it covariantly.
    """

    @abstractmethod
    def build(self, dims: DataDimensions) -> torch.nn.Module:
        """Return the module sized for ``dims``."""
