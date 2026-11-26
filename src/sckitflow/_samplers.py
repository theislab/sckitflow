"""Portable configs for the samplers a flow method draws time and noise with.

Each config binds ``fn`` (``(shape, *, generator, device, dtype) -> Tensor``) as a ``ClassVar`` and its
fields as keyword arguments, like :class:`sckitflow.core.coupling.CouplingConfig`, so a spec never holds
a function. Time and noise samplers are separate families, so one cannot fill the other's slot.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import ClassVar

import torch
from modelbinder.registry import Component, component

__all__ = [
    "SamplerConfig",
    "TimeSamplerConfig",
    "NoiseSamplerConfig",
    "UniformTimeSamplerConfig",
    "IsotropicGaussianNoiseSamplerConfig",
]


class SamplerConfig(Component):
    """Shared base of the sampler families. ``build`` returns ``fn`` with the fields bound."""

    fn: ClassVar[Callable[..., torch.Tensor]]

    def build(self) -> Callable[..., torch.Tensor]:
        kwargs = dict(self)
        if not kwargs:
            return type(self).fn
        return partial(type(self).fn, **kwargs)


class TimeSamplerConfig(SamplerConfig):
    """Family base for time samplers, drawing in ``[0, 1)``."""


class NoiseSamplerConfig(SamplerConfig):
    """Family base for noise samplers, drawing the source distribution."""


def _uniform(
    shape: tuple[int, ...],
    *,
    generator: torch.Generator,
    device: torch.types.Device = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Uniform samples in ``[0, 1)``, drawn from ``generator``."""
    return torch.rand(shape, generator=generator, device=device, dtype=dtype)


def _standard_normal(
    shape: tuple[int, ...],
    *,
    generator: torch.Generator,
    device: torch.types.Device = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Standard normal samples, drawn from ``generator``."""
    return torch.randn(shape, generator=generator, device=device, dtype=dtype)


@component("time_sampler.uniform")
class UniformTimeSamplerConfig(TimeSamplerConfig):
    """Uniform time samples in ``[0, 1)``."""

    fn = _uniform


@component("noise_sampler.isotropic_gaussian")
class IsotropicGaussianNoiseSamplerConfig(NoiseSamplerConfig):
    """Standard normal noise samples — the CFM source-distribution default."""

    fn = _standard_normal
