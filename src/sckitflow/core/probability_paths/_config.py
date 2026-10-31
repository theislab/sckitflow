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
    sigma: float
    sampler_cfg: NoiseSamplerConfig | None = None

    def build(self) -> BasePath:
        raise NotImplementedError


@component("paths.two_sided.linear", builds=LinearTwoSidedPath)
class LinearTwoSidedPathConfig(PathConfig):
    def build(self) -> LinearTwoSidedPath:
        return LinearTwoSidedPath(self.sigma, sampler_cfg=self.sampler_cfg)


@component("paths.two_sided.trig", builds=TrigTwoSidedPath)
class TrigTwoSidedPathConfig(PathConfig):
    def build(self) -> TrigTwoSidedPath:
        return TrigTwoSidedPath(self.sigma, sampler_cfg=self.sampler_cfg)


@component("paths.two_sided.enc_dec", builds=EncDecTwoSidedPath)
class EncDecTwoSidedPathConfig(PathConfig):
    def build(self) -> EncDecTwoSidedPath:
        return EncDecTwoSidedPath(self.sigma, sampler_cfg=self.sampler_cfg)


@component("paths.one_sided.linear", builds=LinearOneSidedPath)
class LinearOneSidedPathConfig(PathConfig):
    def build(self) -> LinearOneSidedPath:
        return LinearOneSidedPath(self.sigma, sampler_cfg=self.sampler_cfg)


@component("paths.one_sided.trig", builds=TrigOneSidedPath)
class TrigOneSidedPathConfig(PathConfig):
    def build(self) -> TrigOneSidedPath:
        return TrigOneSidedPath(self.sigma, sampler_cfg=self.sampler_cfg)


@component("paths.one_sided.vp", builds=VPOneSidedPath)
class VPOneSidedPathConfig(PathConfig):
    def build(self) -> VPOneSidedPath:
        return VPOneSidedPath(self.sigma, sampler_cfg=self.sampler_cfg)
