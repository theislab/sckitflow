from __future__ import annotations

from abc import abstractmethod

from modelbinder.registry import Component, component
from pydantic import Field, model_validator

from sckitflow._samplers import NoiseSamplerConfig
from sckitflow.core.probability_paths._base import BasePath
from sckitflow.core.probability_paths._probability_paths import (
    EncDecPath,
    LinearPath,
    TrigPath,
)

__all__ = [
    "PathConfig",
    "LinearPathConfig",
    "TrigPathConfig",
    "EncDecPathConfig",
]


class PathConfig(Component):
    r"""Family base for probability path configurations.

    Holds the parameters shared by every probability path — the noise strength :math:`\sigma` and an optional
    noise-sampler configuration. Each member builds its own path.

    Concrete members are registered under ``probability_path.*`` keys through :func:`modelbinder.registry.component`. A
    :class:`PathConfig` on its own is a *family base* and is rejected at construction time; instantiate a
    concrete member instead.

    :param sigma: Finite non-negative scalar noise strength of the path, scaling its standard deviation
        :math:`\sigma_t`.

    :param sampler_cfg: Noise-sampler configuration. Required when ``sigma > 0``.
    """

    sigma: float = Field(default=0.0, ge=0)
    sampler_cfg: NoiseSamplerConfig | None = None

    @model_validator(mode="after")
    def _noise_needs_sampler(self) -> PathConfig:
        if self.sigma > 0 and self.sampler_cfg is None:
            raise ValueError("Positive sigma requires a sampler_cfg.")
        return self

    @abstractmethod
    def build(self) -> BasePath:
        r"""Builds the probability path described by this config."""


@component("probability_path.linear")
class LinearPathConfig(PathConfig):
    r"""Config for :class:`~LinearPath`.

    Registered under ``probability_path.linear``. The built path interpolates linearly between
    :math:`\boldsymbol{x}_0` and :math:`\boldsymbol{x}_1` with a :math:`\sigma \sqrt{t (1 - t)}` noise schedule.
    """

    def build(self) -> LinearPath:
        return LinearPath(self.sigma, sampler_cfg=self.sampler_cfg)


@component("probability_path.trig")
class TrigPathConfig(PathConfig):
    r"""Config for :class:`~TrigPath`.

    Registered under ``probability_path.trig``. The built path uses :math:`\cos`/:math:`\sin` interpolation with a
    :math:`\sigma \sqrt{t (1 - t)}` noise schedule.
    """

    def build(self) -> TrigPath:
        return TrigPath(self.sigma, sampler_cfg=self.sampler_cfg)


@component("probability_path.enc_dec")
class EncDecPathConfig(PathConfig):
    r"""Config for :class:`~EncDecPath`.

    Registered under ``probability_path.enc_dec``. The built path uses the piecewise :math:`\cos^2(\pi t)`
    encoder/decoder schedule with a :math:`\sigma \sin^2(\pi t)` noise profile. ``sigma`` must be positive.
    """

    sigma: float = Field(gt=0)

    def build(self) -> EncDecPath:
        return EncDecPath(self.sigma, sampler_cfg=self.sampler_cfg)
