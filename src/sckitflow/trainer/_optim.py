from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from pydantic import field_validator
from scfit.registry import Component, component

__all__ = ["OptimizerConfig"]


@component("optimizer.torch", builds=torch.optim.Optimizer)
class OptimizerConfig(Component):
    """A ``torch.optim`` optimizer by class name, e.g. ``OptimizerConfig(name="Adam", kwargs={"lr": 1e-4})``."""

    name: str = "Adam"
    """A class in ``torch.optim``."""
    kwargs: dict[str, Any] = {}

    @field_validator("name")
    @classmethod
    def _is_optimizer(cls, name: str) -> str:
        if not isinstance(getattr(torch.optim, name, None), type) or not issubclass(
            getattr(torch.optim, name), torch.optim.Optimizer
        ):
            raise ValueError(f"torch.optim has no optimizer {name!r}.")
        return name

    def build(self, params: Iterable[torch.nn.Parameter]) -> torch.optim.Optimizer:
        return getattr(torch.optim, self.name)(params, **self.kwargs)
