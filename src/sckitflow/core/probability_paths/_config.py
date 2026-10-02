"""Portable configs for the probability paths.

A path is only its parameters. A stochastic path draws its noise from the
``generator`` passed to each :meth:`~BaseProbabilityPath.compute_xt` call, so no
config holds a seed.
"""

from __future__ import annotations

from scfit.registry import Component, component

from sckitflow.core.probability_paths._probability_paths import (
    BaseProbabilityPath,
    LinearDiracProbabilityPath,
    LinearGaussianProbabilityPath,
    SchrodingerBridgeProbabilityPath,
    VariancePreservingDiracProbabilityPath,
)

__all__ = [
    "ProbabilityPathConfig",
    "LinearDiracProbabilityPathConfig",
    "LinearGaussianProbabilityPathConfig",
    "SchrodingerBridgeProbabilityPathConfig",
    "VariancePreservingDiracProbabilityPathConfig",
]


class ProbabilityPathConfig(Component):
    """Family base for the probability paths.

    :param sigma: The path's noise scale. Required, except on the deterministic (Dirac) paths, where it defaults to 0.
    """

    sigma: float

    def build(self) -> BaseProbabilityPath:
        raise NotImplementedError


@component("probability_path.linear_dirac", builds=LinearDiracProbabilityPath)
class LinearDiracProbabilityPathConfig(ProbabilityPathConfig):
    """Straight-line interpolation to a Dirac target. Deterministic."""

    sigma: float = 0.0

    def build(self) -> LinearDiracProbabilityPath:
        return LinearDiracProbabilityPath(sigma=self.sigma)


@component("probability_path.linear_gaussian", builds=LinearGaussianProbabilityPath)
class LinearGaussianProbabilityPathConfig(ProbabilityPathConfig):
    """Straight-line interpolation with Gaussian noise."""

    def build(self) -> LinearGaussianProbabilityPath:
        return LinearGaussianProbabilityPath(sigma=self.sigma)


@component("probability_path.schrodinger_bridge", builds=SchrodingerBridgeProbabilityPath)
class SchrodingerBridgeProbabilityPathConfig(ProbabilityPathConfig):
    """Schrödinger-bridge path.

    :param eps: The bridge's entropic regularization.
    """

    eps: float = 1e-35

    def build(self) -> SchrodingerBridgeProbabilityPath:
        return SchrodingerBridgeProbabilityPath(sigma=self.sigma, eps=self.eps)


@component("probability_path.variance_preserving_dirac", builds=VariancePreservingDiracProbabilityPath)
class VariancePreservingDiracProbabilityPathConfig(ProbabilityPathConfig):
    """Variance-preserving path to a Dirac target. Deterministic."""

    sigma: float = 0.0

    def build(self) -> VariancePreservingDiracProbabilityPath:
        return VariancePreservingDiracProbabilityPath(sigma=self.sigma)
