"""Potentials used by more than one surrogate, as components so a surrogate's spec can name them."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import ClassVar

import torch
from scfit.registry import Component, component

__all__ = [
    "Readout",
    "Potential",
    "PotentialConfig",
    "SquaredErrorPotentialConfig",
    "CrossEntropyPotentialConfig",
    "squared_error",
    "cross_entropy_probs",
]

Readout = Callable[[torch.Tensor], torch.Tensor]
"""A per-cell map ``(..., D) -> (..., D')``, e.g. a cell-type classifier's probabilities."""

Potential = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
"""``(y, ystar)``, shapes ``(..., N, D')`` and ``(D',)`` -> ``(..., N)``, one potential per row."""


def squared_error(y: torch.Tensor, ystar: torch.Tensor) -> torch.Tensor:
    """The squared error, summed over the response dimensions."""
    return ((y - ystar) ** 2).sum(-1)


def cross_entropy_probs(y: torch.Tensor, ystar: torch.Tensor) -> torch.Tensor:
    """``-Σ ystar log y`` for a response ``y`` of class probabilities, e.g. a mean cell-type composition.

    Unlike :func:`torch.nn.functional.cross_entropy`, both arguments are probabilities, not logits and labels.
    """
    return -(ystar * torch.log(y.clamp_min(torch.finfo(y.dtype).tiny))).sum(-1)


class PotentialConfig(Component):
    """Family base for the potentials. ``build`` binds every field to ``fn`` as keyword arguments."""

    fn: ClassVar[Callable[..., torch.Tensor]]

    def build(self) -> Potential:
        return partial(type(self).fn, **dict(self))


@component("surrogate.potential.squared_error")
class SquaredErrorPotentialConfig(PotentialConfig):
    """See `squared_error`."""

    fn = squared_error


@component("surrogate.potential.cross_entropy")
class CrossEntropyPotentialConfig(PotentialConfig):
    """See `cross_entropy_probs`."""

    fn = cross_entropy_probs
