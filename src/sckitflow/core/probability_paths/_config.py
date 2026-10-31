from __future__ import annotations

from scfit.registry import Component, component

from sckitflow._samplers import NoiseSamplerConfig
from sckitflow.core.probability_paths._base import BasePath
from sckitflow.core.probability_paths._probability_paths import (
    EncDecTwoSidedPath,
    LinearOneSidedPath,
    LinearTwoSidedPath,
    TrigOneSidedPath,
    TrigTwoSidedPath,
    VPOneSidedPath,
)

__all__ = [
    "PathConfig",
    "LinearTwoSidedPathConfig",
    "TrigTwoSidedPathConfig",
    "EncDecTwoSidedPathConfig",
    "LinearOneSidedPathConfig",
    "TrigOneSidedPathConfig",
    "VPOneSidedPathConfig",
]


class PathConfig(Component):
    r"""Family base for probability path configurations.

    Holds the parameters shared by every probability path — the noise
    strength :math:`\sigma` and an optional noise-sampler configuration —
    and exposes a single :meth:`build` entry point that materialises the
    concrete :class:`~BasePath` described by the config.

    Concrete members are registered under ``paths.*`` keys through the
    :func:`scfit.registry.component` decorator; each member pins one
    :class:`~BasePath` subclass via its ``builds=`` argument.

    A :class:`PathConfig` on its own is a *family base* and is rejected at
    construction time; instantiate a concrete member instead.

    :param sigma: Non-negative scalar noise strength of the path. Must be
        provided on every concrete config.
    :type sigma: class: `float`

    :param sampler_cfg: Optional noise-sampler configuration. Required when
        ``sigma > 0`` and ignored for deterministic paths.
    :type sampler_cfg: class: `NoiseSamplerConfig` | None
    """

    sigma: float
    sampler_cfg: NoiseSamplerConfig | None = None

    def build(self) -> BasePath:
        r"""Builds the concrete probability path described by this config.

        :returns: The configured probability path.
        :rtype: class: `BasePath`

        :raises NotImplementedError: Always; overridden by concrete members.
        """
        raise NotImplementedError


@component("paths.two_sided.linear", builds=LinearTwoSidedPath)
class LinearTwoSidedPathConfig(PathConfig):
    r"""Config for :class:`~LinearTwoSidedPath`.

    Registered under ``paths.two_sided.linear``. The built path interpolates
    linearly between :math:`\boldsymbol{x}_0` and :math:`\boldsymbol{x}_1`
    with a :math:`\sqrt{\sigma \, t (1 - t)}` noise schedule.
    """

    def build(self) -> LinearTwoSidedPath:
        r"""Builds a :class:`~LinearTwoSidedPath` from this config.

        :returns: The configured :class:`~LinearTwoSidedPath`.
        :rtype: class: `LinearTwoSidedPath`
        """
        return LinearTwoSidedPath(self.sigma, sampler_cfg=self.sampler_cfg)


@component("paths.two_sided.trig", builds=TrigTwoSidedPath)
class TrigTwoSidedPathConfig(PathConfig):
    r"""Config for :class:`~TrigTwoSidedPath`.

    Registered under ``paths.two_sided.trig``. The built path uses
    :math:`\cos`/:math:`\sin` interpolation with a
    :math:`\sqrt{\sigma \, t (1 - t)}` noise schedule.
    """

    def build(self) -> TrigTwoSidedPath:
        r"""Builds a :class:`~TrigTwoSidedPath` from this config.

        :returns: The configured :class:`~TrigTwoSidedPath`.
        :rtype: class: `TrigTwoSidedPath`
        """
        return TrigTwoSidedPath(self.sigma, sampler_cfg=self.sampler_cfg)


@component("paths.two_sided.enc_dec", builds=EncDecTwoSidedPath)
class EncDecTwoSidedPathConfig(PathConfig):
    r"""Config for :class:`~EncDecTwoSidedPath`.

    Registered under ``paths.two_sided.enc_dec``. The built path uses the
    piecewise :math:`\cos^2(\pi t)` encoder/decoder schedule with a
    :math:`\sigma \sin^2(\pi t)` noise profile.
    """

    def build(self) -> EncDecTwoSidedPath:
        r"""Builds an :class:`~EncDecTwoSidedPath` from this config.

        :returns: The configured :class:`~EncDecTwoSidedPath`.
        :rtype: class: `EncDecTwoSidedPath`
        """
        return EncDecTwoSidedPath(self.sigma, sampler_cfg=self.sampler_cfg)


@component("paths.one_sided.linear", builds=LinearOneSidedPath)
class LinearOneSidedPathConfig(PathConfig):
    r"""Config for :class:`~LinearOneSidedPath`.

    Registered under ``paths.one_sided.linear``. The built path has
    :math:`\alpha_t \equiv 0`, :math:`\beta_t = t`, and a linearly decaying
    noise schedule :math:`\sigma_t = 1 - t`.
    """

    def build(self) -> LinearOneSidedPath:
        r"""Builds a :class:`~LinearOneSidedPath` from this config.

        :returns: The configured :class:`~LinearOneSidedPath`.
        :rtype: class: `LinearOneSidedPath`
        """
        return LinearOneSidedPath(self.sigma, sampler_cfg=self.sampler_cfg)


@component("paths.one_sided.trig", builds=TrigOneSidedPath)
class TrigOneSidedPathConfig(PathConfig):
    r"""Config for :class:`~TrigOneSidedPath`.

    Registered under ``paths.one_sided.trig``. The built path has
    :math:`\alpha_t \equiv 0`,
    :math:`\beta_t = \sin(\tfrac{\pi}{2} t)`, and a variance-preserving
    noise schedule :math:`\sigma_t = \cos(\tfrac{\pi}{2} t)`.
    """

    def build(self) -> TrigOneSidedPath:
        r"""Builds a :class:`~TrigOneSidedPath` from this config.

        :returns: The configured :class:`~TrigOneSidedPath`.
        :rtype: class: `TrigOneSidedPath`
        """
        return TrigOneSidedPath(self.sigma, sampler_cfg=self.sampler_cfg)


@component("paths.one_sided.vp", builds=VPOneSidedPath)
class VPOneSidedPathConfig(PathConfig):
    r"""Config for :class:`~VPOneSidedPath`.

    Registered under ``paths.one_sided.vp``. The built path has
    :math:`\alpha_t \equiv 0`, :math:`\beta_t = t`, and a variance-
    preserving noise schedule :math:`\sigma_t = \sqrt{1 - t^2}`.
    """

    def build(self) -> VPOneSidedPath:
        r"""Builds a :class:`~VPOneSidedPath` from this config.

        :returns: The configured :class:`~VPOneSidedPath`.
        :rtype: class: `VPOneSidedPath`
        """
        return VPOneSidedPath(self.sigma, sampler_cfg=self.sampler_cfg)
