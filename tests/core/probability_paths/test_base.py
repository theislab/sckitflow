"""Tests for BasePath on a minimal analytic path; the concrete paths are in test_probability_paths."""

import pytest
import torch

from sckitflow import IsotropicGaussianNoiseSamplerConfig
from sckitflow.core.probability_paths._base import BasePath


class _LinearTestPath(BasePath):
    """``alpha_t = 1 - t``, ``beta_t = t``, ``sigma_t = sigma t(1 - t)``."""

    def compute_alpha_t(self, t: torch.Tensor) -> torch.Tensor:
        return 1 - t

    def compute_beta_t(self, t: torch.Tensor) -> torch.Tensor:
        return t

    def compute_sigma_t(self, t: torch.Tensor) -> torch.Tensor:
        return self.sigma * t * (1 - t)


def _recording_sampler_cfg(noise: torch.Tensor, calls: list) -> IsotropicGaussianNoiseSamplerConfig:
    """A sampler config whose sampler records its calls and returns ``noise``."""

    def sampler(shape, *, generator, device=None, dtype=None):
        calls.append(shape)
        return noise.to(device=device, dtype=dtype)

    class _RecordingSamplerConfig(IsotropicGaussianNoiseSamplerConfig):
        def build(self):
            return sampler

    return _RecordingSamplerConfig()


@pytest.mark.parametrize("sigma", [-1.0, float("inf"), float("nan")])
def test_invalid_sigma_raises(sigma) -> None:
    with pytest.raises(ValueError, match="finite non-negative"):
        _LinearTestPath(sigma)


def test_positive_sigma_without_sampler_raises() -> None:
    with pytest.raises(ValueError, match="Positive sigma requires a sampler"):
        _LinearTestPath(1.0)


class TestComputeXt:
    t = torch.tensor([[0.5], [0.25]], dtype=torch.float64)
    x0 = torch.tensor([[1.0, -1.0], [2.0, 0.0]], dtype=torch.float64)
    x1 = torch.tensor([[3.0, 1.0], [-2.0, 4.0]], dtype=torch.float64)
    mu = torch.tensor([[2.0, 0.0], [1.0, 1.0]], dtype=torch.float64)

    def test_deterministic_returns_mu(self) -> None:
        calls = []
        path = _LinearTestPath(0.0, sampler_cfg=_recording_sampler_cfg(torch.ones(2, 2), calls))
        torch.testing.assert_close(path.compute_xt(self.t, self.x0, self.x1, generator=torch.Generator()), self.mu)
        assert calls == []

    def test_draws_noise_from_sampler_cfg(self) -> None:
        calls = []
        noise = torch.tensor([[1.0, 2.0], [-1.0, 4.0]], dtype=torch.float64)
        path = _LinearTestPath(2.0, sampler_cfg=_recording_sampler_cfg(noise, calls))
        xt = path.compute_xt(self.t, self.x0, self.x1, generator=torch.Generator())
        # sigma_t = 2 t (1 - t): 0.5 at t = 0.5 and 0.375 at t = 0.25
        torch.testing.assert_close(xt, self.mu + torch.tensor([[0.5], [0.375]], dtype=torch.float64) * noise)
        assert calls == [(2, 2)]


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
        # ``sigma_t(0) == sigma_t(1) == 0``; without the eps guard the result would be ``inf``.
        path = _LinearTestPath(1.0, sampler_cfg=IsotropicGaussianNoiseSamplerConfig())
        t = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
        x0 = torch.randn(2, 4, dtype=torch.float64)
        x1 = torch.randn(2, 4, dtype=torch.float64)
        xt = torch.randn(2, 4, dtype=torch.float64)
        # coeff is guarded to 0 => ut collapses to dot_mu_t
        torch.testing.assert_close(path.compute_ut(t, xt, x0, x1), -x0 + x1)
