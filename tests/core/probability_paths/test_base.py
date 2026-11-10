"""Tests for BasePath and OneSidedPath on minimal analytic paths; the concrete paths are in test_probability_paths."""

import pytest
import torch

from sckitflow import IsotropicGaussianNoiseSamplerConfig
from sckitflow.core.probability_paths._base import BasePath, OneSidedPath


class _LinearTestPath(BasePath):
    """``alpha_t = 1 - t``, ``beta_t = t``, ``sigma_t = t(1 - t)``."""

    def compute_alpha_t(self, t: torch.Tensor) -> torch.Tensor:
        return 1 - t

    def compute_beta_t(self, t: torch.Tensor) -> torch.Tensor:
        return t

    def compute_sigma_t(self, t: torch.Tensor) -> torch.Tensor:
        return t * (1 - t)


class TestBasePathInit:
    def test_negative_sigma_raises(self) -> None:
        with pytest.raises(ValueError, match="Argument sigma should be a positive float"):
            _LinearTestPath(-1.0)

    def test_positive_sigma_without_sampler_raises(self) -> None:
        with pytest.raises(ValueError, match="Positive sigma requires a sampler"):
            _LinearTestPath(1.0)

    def test_zero_sigma_without_sampler_ok(self) -> None:
        path = _LinearTestPath(0.0)
        assert path.sigma == 0.0
        assert path._sampler is None

    def test_positive_sigma_with_sampler_builds_sampler(self) -> None:
        path = _LinearTestPath(0.3, sampler_cfg=IsotropicGaussianNoiseSamplerConfig())
        assert path.sigma == 0.3
        assert path._sampler is not None


class TestAbstractness:
    def test_basepath_is_abstract(self) -> None:
        with pytest.raises(TypeError, match="abstract"):
            BasePath(0.0)

    def test_one_sided_path_is_abstract(self) -> None:
        # ``compute_beta_t`` and ``compute_sigma_t`` are still abstract.
        with pytest.raises(TypeError, match="abstract"):
            OneSidedPath(1.0)


class TestComputeXt:
    def test_deterministic_returns_mu(self) -> None:
        path = _LinearTestPath(0.0)
        t = torch.tensor([[0.3], [0.7]], dtype=torch.float64)
        x0 = torch.randn(2, 4, dtype=torch.float64)
        x1 = torch.randn(2, 4, dtype=torch.float64)
        xt = path.compute_xt(t, x0, x1, generator=torch.Generator())
        torch.testing.assert_close(xt, (1 - t) * x0 + t * x1)


class TestComputeUt:
    def test_deterministic_returns_dot_mu(self) -> None:
        path = _LinearTestPath(0.0)
        t = torch.tensor([[0.3], [0.7]], dtype=torch.float64)
        x0 = torch.randn(2, 4, dtype=torch.float64)
        x1 = torch.randn(2, 4, dtype=torch.float64)
        ut = path.compute_ut(t, torch.zeros_like(x0), x0, x1)
        torch.testing.assert_close(ut, -x0 + x1)

    def test_matches_manual_formula(self) -> None:
        path = _LinearTestPath(1.0, sampler_cfg=IsotropicGaussianNoiseSamplerConfig())
        t = torch.tensor([[0.3], [0.7]], dtype=torch.float64)
        x0 = torch.randn(2, 4, dtype=torch.float64)
        x1 = torch.randn(2, 4, dtype=torch.float64)
        xt = torch.randn(2, 4, dtype=torch.float64)

        ut = path.compute_ut(t, xt, x0, x1)

        mu_t = (1 - t) * x0 + t * x1
        expected = (-x0 + x1) + ((1 - 2 * t) / (t * (1 - t))) * (xt - mu_t)
        torch.testing.assert_close(ut, expected)

    def test_eps_guard_at_endpoints(self) -> None:
        # ``sigma_t(0) == sigma_t(1) == 0``; without the eps guard the result would be ``nan``.
        path = _LinearTestPath(1.0, sampler_cfg=IsotropicGaussianNoiseSamplerConfig())
        t = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
        x0 = torch.randn(2, 4, dtype=torch.float64)
        x1 = torch.randn(2, 4, dtype=torch.float64)
        xt = torch.randn(2, 4, dtype=torch.float64)
        # coeff is guarded to 0 => ut collapses to dot_mu_t
        torch.testing.assert_close(path.compute_ut(t, xt, x0, x1), -x0 + x1)
