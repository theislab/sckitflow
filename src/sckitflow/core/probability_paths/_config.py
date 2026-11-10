from __future__ import annotations

from typing import ClassVar

from scfit.registry import Component, component

from sckitflow._samplers import NoiseSamplerConfig
from sckitflow.core.probability_paths._base import BasePath
from sckitflow.core.probability_paths._probability_paths import (
    EncDecTwoSidedPath,
    LinearTwoSidedPath,
    TrigTwoSidedPath,
)

__all__ = [
    "PathConfig",
    "LinearTwoSidedPathConfig",
    "TrigTwoSidedPathConfig",
    "EncDecTwoSidedPathConfig",
]


class PathConfig(Component):
    r"""Family base for probability path configurations.

    Holds the parameters shared by every probability path — the noise strength :math:`\sigma` and an optional
    noise-sampler configuration — and builds the path its member names in :attr:`path_cls`.

    Concrete members are registered under ``paths.*`` keys through :func:`scfit.registry.component`. A
    :class:`PathConfig` on its own is a *family base* and is rejected at construction time; instantiate a
    concrete member instead.

    :param sigma: Non-negative scalar noise strength of the path, scaling its standard deviation :math:`\sigma_t`.
        One-sided paths take ``1``.

    :param sampler_cfg: Optional noise-sampler configuration. Required by two-sided paths when ``sigma > 0``;
        one-sided paths take none, their noise is the flow method's latent.
    """

    path_cls: ClassVar[type[BasePath]]
    """The path this member builds."""

    sigma: float
    sampler_cfg: NoiseSamplerConfig | None = None

    def build(self) -> BasePath:
        r"""Builds the probability path described by this config."""
        return type(self).path_cls(self.sigma, sampler_cfg=self.sampler_cfg)


@component("paths.two_sided.linear", builds=LinearTwoSidedPath)
class LinearTwoSidedPathConfig(PathConfig):
    r"""Config for :class:`~LinearTwoSidedPath`.

    Registered under ``paths.two_sided.linear``. The built path interpolates linearly between
    :math:`\boldsymbol{x}_0` and :math:`\boldsymbol{x}_1` with a :math:`\sigma \sqrt{t (1 - t)}` noise schedule.
    """

    path_cls = LinearTwoSidedPath


@component("paths.two_sided.trig", builds=TrigTwoSidedPath)
class TrigTwoSidedPathConfig(PathConfig):
    r"""Config for :class:`~TrigTwoSidedPath`.

    Registered under ``paths.two_sided.trig``. The built path uses :math:`\cos`/:math:`\sin` interpolation with a
    :math:`\sigma \sqrt{t (1 - t)}` noise schedule.
    """

    path_cls = TrigTwoSidedPath


@component("paths.two_sided.enc_dec", builds=EncDecTwoSidedPath)
class EncDecTwoSidedPathConfig(PathConfig):
    r"""Config for :class:`~EncDecTwoSidedPath`.

    Registered under ``paths.two_sided.enc_dec``. The built path uses the piecewise :math:`\cos^2(\pi t)`
    encoder/decoder schedule with a :math:`\sigma \sin^2(\pi t)` noise profile.
    """

    path_cls = EncDecTwoSidedPath
