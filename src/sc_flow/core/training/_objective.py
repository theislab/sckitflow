"""The training seam: a model (torch weights) + an objective (computes the loss).

"How a batch becomes a scalar loss" is a small, registerable :class:`Objective`. The weights always live
in a torch ``nn.Module`` (the *model*); the objective decides *where the numerics run* — natively in
torch, or in JAX via the DLPack bridge with the torch weights mirrored per step. Both are trained by the
one :class:`~sc_flow.core.training._harness.TrainingModule`, so "torch vs JAX compute" is a
one-line objective swap rather than a second LightningModule.

This module is the generic seam (base class + registries); concrete objectives live in the
flow-matching layer (:mod:`sc_flow.flow`) and register themselves on import.
"""

from __future__ import annotations

import abc
from collections.abc import Callable
from typing import Any

import torch

__all__ = [
    "Objective",
    "register_objective",
    "register_architecture",
    "build_objective",
    "build_architecture",
    "OBJECTIVE_REGISTRY",
    "ARCHITECTURE_REGISTRY",
]


class Objective(abc.ABC):
    """Computes a scalar training loss for a model on a batch.

    The single seam the harness calls each step. Implementations own *where* the numerics run (torch or
    JAX) and how they read the batch; the harness only sees the returned loss (whose gradient must flow to
    ``model``'s torch parameters) and logs.
    """

    @abc.abstractmethod
    def compute_loss(self, model: torch.nn.Module, batch: Any) -> tuple[torch.Tensor, dict[str, Any]]:
        """Return ``(loss, logs)``. ``loss.backward()`` must reach ``model.parameters()``."""


# Registries so third-party architectures / objectives are discoverable by name.
# An architecture builder returns a torch ``nn.Module`` (the weights); an objective builder returns an
# :class:`Objective`.
ARCHITECTURE_REGISTRY: dict[str, Callable[..., torch.nn.Module]] = {}
OBJECTIVE_REGISTRY: dict[str, Callable[..., Objective]] = {}


def register_architecture(name: str) -> Callable[[Callable[..., torch.nn.Module]], Callable[..., torch.nn.Module]]:
    """Register a model (architecture) builder under ``name``.

    The builder returns a torch ``nn.Module`` holding the weights. Lets someone
    ``register_architecture("my-net")`` and select it by config without editing the framework.
    """

    def deco(builder: Callable[..., torch.nn.Module]) -> Callable[..., torch.nn.Module]:
        if name in ARCHITECTURE_REGISTRY:
            raise ValueError(f"Architecture {name!r} already registered.")
        ARCHITECTURE_REGISTRY[name] = builder
        return builder

    return deco


def register_objective(name: str) -> Callable[[Callable[..., Objective]], Callable[..., Objective]]:
    """Register an :class:`Objective` builder under ``name`` (e.g. ``"fm-linear"``, ``"otfm"``, ``"genot"``)."""

    def deco(builder: Callable[..., Objective]) -> Callable[..., Objective]:
        if name in OBJECTIVE_REGISTRY:
            raise ValueError(f"Objective {name!r} already registered.")
        OBJECTIVE_REGISTRY[name] = builder
        return builder

    return deco


def build_architecture(name: str, *args: Any, **kwargs: Any) -> torch.nn.Module:
    """Instantiate a registered architecture by name."""
    if name not in ARCHITECTURE_REGISTRY:
        raise KeyError(f"Architecture {name!r} not registered. Available: {sorted(ARCHITECTURE_REGISTRY)}.")
    return ARCHITECTURE_REGISTRY[name](*args, **kwargs)


def build_objective(name: str, *args: Any, **kwargs: Any) -> Objective:
    """Instantiate a registered objective by name.

    Concrete objectives live in :mod:`sc_flow.flow` and register on import; import that layer (the
    :class:`~sc_flow.flow` package does this) before building by name.
    """
    if name not in OBJECTIVE_REGISTRY:
        raise KeyError(f"Objective {name!r} not registered. Available: {sorted(OBJECTIVE_REGISTRY)}.")
    return OBJECTIVE_REGISTRY[name](*args, **kwargs)
