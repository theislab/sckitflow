"""Readouts and potentials used by more than one surrogate, as components so a surrogate's spec can name them."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import ClassVar

import torch
from modelbinder.registry import Component, component

from sckitflow.core._types import PredictionData

__all__ = [
    "Readout",
    "Potential",
    "ReadoutConfig",
    "PotentialConfig",
    "MeanSamplesReadoutConfig",
    "SquaredErrorPotentialConfig",
    "mean_samples",
    "squared_error",
]

Readout = Callable[[PredictionData], torch.Tensor]
"""A prediction -> ``(N, D)``, one response per condition."""

Potential = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
"""``(y, ystar)``, shapes ``(N, D')`` and ``(D',)`` -> ``(N,)``, one potential per condition."""


def mean_samples(pred_data: PredictionData) -> torch.Tensor:
    """The mean of ``raw_samples`` over every axis before the conditions': the Monte Carlo mean response."""
    samples = pred_data.raw_samples
    return samples.reshape(-1, *samples.shape[-2:]).mean(0)


def squared_error(y: torch.Tensor, ystar: torch.Tensor) -> torch.Tensor:
    """The squared error, summed over the response dimensions."""
    return ((y - ystar) ** 2).sum(-1)


class ReadoutConfig(Component):
    """Family base for the readouts. ``build`` binds every field to ``fn`` as keyword arguments."""

    fn: ClassVar[Callable[..., torch.Tensor]]

    def build(self) -> Readout:
        return partial(type(self).fn, **dict(self))


class PotentialConfig(Component):
    """Family base for the potentials. ``build`` binds every field to ``fn`` as keyword arguments."""

    fn: ClassVar[Callable[..., torch.Tensor]]

    def build(self) -> Potential:
        return partial(type(self).fn, **dict(self))


@component("surrogate.readout.mean_samples")
class MeanSamplesReadoutConfig(ReadoutConfig):
    """See `mean_samples`."""

    fn = mean_samples


@component("surrogate.potential.squared_error")
class SquaredErrorPotentialConfig(PotentialConfig):
    """See `squared_error`."""

    fn = squared_error
