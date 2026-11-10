from __future__ import annotations

from typing import ClassVar

from pydantic import Field, model_validator
from scfit.registry import Component, component

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
    noise-sampler configuration — and builds the path its member names in :attr:`path_cls`.

    Concrete members are registered under ``probability_path.*`` keys through :func:`scfit.registry.component`. A
    :class:`PathConfig` on its own is a *family base* and is rejected at construction time; instantiate a
    concrete member instead.

    :param sigma: Finite non-negative scalar noise strength of the path, scaling its standard deviation
        :math:`\sigma_t`.

    :param sampler_cfg: Noise-sampler configuration. Required when ``sigma > 0``.
    """

    path_cls: ClassVar[type[BasePath]]
    """The path this member builds."""

    sigma: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    sampler_cfg: NoiseSamplerConfig | None = None

    @model_validator(mode="after")
    def _noise_needs_sampler(self) -> PathConfig:
        if self.sigma > 0 and self.sampler_cfg is None:
            raise ValueError("Positive sigma requires a sampler_cfg.")
        return self

    def build(self) -> BasePath:
        r"""Builds the probability path described by this config."""
        return type(self).path_cls(self.sigma, sampler_cfg=self.sampler_cfg)


@component("probability_path.linear", builds=LinearPath)
class LinearPathConfig(PathConfig):
    r"""Config for :class:`~LinearPath`.

    Registered under ``probability_path.linear``. The built path interpolates linearly between
    :math:`\boldsymbol{x}_0` and :math:`\boldsymbol{x}_1` with a :math:`\sigma \sqrt{t (1 - t)}` noise schedule.
    """

    path_cls = LinearPath


@component("probability_path.trig", builds=TrigPath)
class TrigPathConfig(PathConfig):
    r"""Config for :class:`~TrigPath`.

    Registered under ``probability_path.trig``. The built path uses :math:`\cos`/:math:`\sin` interpolation with a
    :math:`\sigma \sqrt{t (1 - t)}` noise schedule.
    """

    path_cls = TrigPath


@component("probability_path.enc_dec", builds=EncDecPath)
class EncDecPathConfig(PathConfig):
    r"""Config for :class:`~EncDecPath`.

    Registered under ``probability_path.enc_dec``. The built path uses the piecewise :math:`\cos^2(\pi t)`
    encoder/decoder schedule with a :math:`\sigma \sin^2(\pi t)` noise profile. ``sigma`` must be positive.
    """

    sigma: float = Field(gt=0, allow_inf_nan=False)

    path_cls = EncDecPath
