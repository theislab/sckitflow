from __future__ import annotations

import abc
from collections.abc import Callable
from typing import Any

import torch

__all__ = [
    "Objective",
    "register_objective",
    "build_objective",
    "OBJECTIVE_REGISTRY",
]


class Objective(abc.ABC):
    @abc.abstractmethod
    def compute_loss(self, model: torch.nn.Module, batch: Any) -> tuple[torch.Tensor, dict[str, Any]]: ...


# Registry so third-party objectives are discoverable by name. An objective builder returns an
# :class:`Objective`. (Top-level architectures are reconstructed from their component spec via the
# sc_flow.ComponentRegistry, not a builder-by-name registry.)
OBJECTIVE_REGISTRY: dict[str, Callable[..., Objective]] = {}


def register_objective(name: str) -> Callable[[Callable[..., Objective]], Callable[..., Objective]]:

    def deco(builder: Callable[..., Objective]) -> Callable[..., Objective]:
        if name in OBJECTIVE_REGISTRY:
            raise ValueError(f"Objective {name!r} already registered.")
        OBJECTIVE_REGISTRY[name] = builder
        return builder

    return deco


def build_objective(name: str, *args: Any, **kwargs: Any) -> Objective:
    if name not in OBJECTIVE_REGISTRY:
        raise KeyError(f"Objective {name!r} not registered. Available: {sorted(OBJECTIVE_REGISTRY)}.")
    return OBJECTIVE_REGISTRY[name](*args, **kwargs)
