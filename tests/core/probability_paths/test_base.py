"""Tests for BasePath and OneSidedPath."""

import pytest
import torch

from sckitflow.core.probability_paths._base import BasePath, OneSidedPath

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _FakeSamplerCfg:
    """Minimal stand-in for ``NoiseSamplerConfig`` — standard Gaussian noise."""

    def build(self):
        def sampler(shape, *, generator, device, dtype):
            return torch.randn(shape, generator=generator, device=device, dtype=dtype)

        return sampler


class _FixedSamplerCfg:
    """Sampler that always returns the same epsilon — for JVP tests."""

    def __init__(self, eps: torch.Tensor) -> None:
        self.eps = eps

    def build(self):
        eps = self.eps

        def sampler(shape, *, generator, device, dtype):
            return eps.to(device=device, dtype=dtype)

        return sampler


class _LinearTestPath(BasePath):
    """Concrete path with simple analytic coefficients.

    ``alpha_t = 1 - t``, ``beta_t = t``, ``sigma_t = t(1 - t)``, so
    ``dot_alpha = -1``, ``dot_beta = 1``, ``dot_sigma = 1 - 2t``.
    """

    def compute_alpha_t(self, t: torch.Tensor) -> torch.Tensor:
        return 1 - t

    def compute_beta_t(self, t: torch.Tensor) -> torch.Tensor:
        return t

    def compute_sigma_t(self, t: torch.Tensor) -> torch.Tensor:
        return t * (1 - t)


class _ConcreteOneSidedPath(OneSidedPath):
    def compute_beta_t(self, t: torch.Tensor) -> torch.Tensor:
        return t

    def compute_sigma_t(self, t: torch.Tensor) -> torch.Tensor:
        return 1 - t


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


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
        path = _LinearTestPath(1.0, sampler_cfg=_FakeSamplerCfg())
        assert path.sigma == 1.0
        assert path._sampler is not None

    def test_sigma_property(self) -> None:
        path = _LinearTestPath(0.3, sampler_cfg=_FakeSamplerCfg())
        assert path.sigma == 0.3


# ---------------------------------------------------------------------------
# Abstractness
# ---------------------------------------------------------------------------


class TestAbstractness:
    def test_basepath_is_abstract(self) -> None:
        with pytest.raises(TypeError, match="abstract"):
            BasePath(0.0)

    def test_one_sided_path_is_abstract(self) -> None:
        # ``compute_beta_t`` and ``compute_sigma_t`` are still abstract.
        with pytest.raises(TypeError, match="abstract"):
            OneSidedPath(0.0)

    def test_one_sided_compute_alpha_is_zero(self) -> None:
        path = _ConcreteOneSidedPath(1.0, sampler_cfg=_FakeSamplerCfg())
        t = torch.linspace(0.0, 1.0, 5)
        assert torch.equal(path.compute_alpha_t(t), torch.zeros_like(t))


# ---------------------------------------------------------------------------
# VJP derivative helpers
# ---------------------------------------------------------------------------


class TestVJPDerivatives:
    def setup_method(self) -> None:
        self.path = _LinearTestPath(1.0, sampler_cfg=_FakeSamplerCfg())
        self.t = torch.linspace(0.1, 0.9, 5, dtype=torch.float64)

    def test_alpha_value_and_dot(self) -> None:
        alpha, dot_alpha = self.path.compute_alpha_t_and_dot_alpha_t(self.t)
        torch.testing.assert_close(alpha, 1 - self.t)
        torch.testing.assert_close(dot_alpha, -torch.ones_like(self.t))

    def test_beta_value_and_dot(self) -> None:
        beta, dot_beta = self.path.compute_beta_t_and_dot_beta_t(self.t)
        torch.testing.assert_close(beta, self.t)
        torch.testing.assert_close(dot_beta, torch.ones_like(self.t))

    def test_sigma_value_and_dot(self) -> None:
        sigma, dot_sigma = self.path.compute_sigma_t_and_dot_sigma_t(self.t)
        torch.testing.assert_close(sigma, self.t * (1 - self.t))
        torch.testing.assert_close(dot_sigma, 1 - 2 * self.t)

    def test_dot_alpha_t(self) -> None:
        torch.testing.assert_close(self.path.compute_dot_alpha_t(self.t), -torch.ones_like(self.t))

    def test_dot_beta_t(self) -> None:
        torch.testing.assert_close(self.path.compute_dot_beta_t(self.t), torch.ones_like(self.t))

    def test_dot_sigma_t(self) -> None:
        torch.testing.assert_close(self.path.compute_dot_sigma_t(self.t), 1 - 2 * self.t)


# ---------------------------------------------------------------------------
# compute_xt
# ---------------------------------------------------------------------------


class TestComputeXt:
    def test_deterministic_returns_mu(self) -> None:
        path = _LinearTestPath(0.0)
        t = torch.tensor([[0.3], [0.7]], dtype=torch.float64)
        x0 = torch.randn(2, 4, dtype=torch.float64)
        x1 = torch.randn(2, 4, dtype=torch.float64)
        xt = path.compute_xt(t, torch.zeros_like(x0), x0, x1, generator=torch.Generator())
        torch.testing.assert_close(xt, (1 - t) * x0 + t * x1)

    def test_shape_matches_x0(self) -> None:
        path = _LinearTestPath(1.0, sampler_cfg=_FakeSamplerCfg())
        t = torch.full((2, 1), 0.5)
        x0 = torch.randn(2, 4, 3)
        x1 = torch.randn(2, 4, 3)
        out = path.compute_xt(t, torch.zeros_like(x0), x0, x1, generator=torch.Generator())
        assert out.shape == x0.shape

    def test_noise_driven_by_generator_only(self) -> None:
        path = _LinearTestPath(1.0, sampler_cfg=_FakeSamplerCfg())
        t = torch.full((4, 1), 0.5)
        x0 = torch.zeros(4, 3)
        x1 = torch.ones(4, 3)
        xt_arg = torch.zeros_like(x0)

        a = path.compute_xt(t, xt_arg, x0, x1, generator=torch.Generator().manual_seed(0))
        b = path.compute_xt(t, xt_arg, x0, x1, generator=torch.Generator().manual_seed(0))
        assert torch.equal(a, b)

        torch.manual_seed(1)  # global RNG must not matter
        c = path.compute_xt(t, xt_arg, x0, x1, generator=torch.Generator().manual_seed(1))
        assert not torch.equal(a, c)

    def test_deterministic_ignores_generator(self) -> None:
        path = _LinearTestPath(0.0)
        t = torch.full((4, 1), 0.5)
        x0 = torch.zeros(4, 3)
        x1 = torch.ones(4, 3)
        a = path.compute_xt(t, torch.zeros_like(x0), x0, x1, generator=torch.Generator().manual_seed(0))
        b = path.compute_xt(t, torch.zeros_like(x0), x0, x1, generator=torch.Generator().manual_seed(123))
        assert torch.equal(a, b)


# ---------------------------------------------------------------------------
# compute_ut
# ---------------------------------------------------------------------------


class TestComputeUt:
    def test_deterministic_returns_dot_mu(self) -> None:
        path = _LinearTestPath(0.0)
        t = torch.tensor([[0.3], [0.7]], dtype=torch.float64)
        x0 = torch.randn(2, 4, dtype=torch.float64)
        x1 = torch.randn(2, 4, dtype=torch.float64)
        ut = path.compute_ut(t, torch.zeros_like(x0), x0, x1, generator=torch.Generator())
        # dot_alpha = -1, dot_beta = 1  =>  -x0 + x1
        torch.testing.assert_close(ut, -x0 + x1)

    def test_matches_manual_formula(self) -> None:
        path = _LinearTestPath(1.0, sampler_cfg=_FakeSamplerCfg())
        t = torch.tensor([[0.3], [0.7]], dtype=torch.float64)
        x0 = torch.randn(2, 4, dtype=torch.float64)
        x1 = torch.randn(2, 4, dtype=torch.float64)
        xt = torch.randn(2, 4, dtype=torch.float64)

        ut = path.compute_ut(t, xt, x0, x1, generator=torch.Generator())

        mu_t = (1 - t) * x0 + t * x1
        dot_mu_t = -x0 + x1
        sigma_t = t * (1 - t)
        dot_sigma_t = 1 - 2 * t
        expected = dot_mu_t + (dot_sigma_t / sigma_t) * (xt - mu_t)
        torch.testing.assert_close(ut, expected)

    def test_eps_guard_at_endpoints(self) -> None:
        # ``sigma_t(0) == sigma_t(1) == 0``; without the eps guard the result
        # would be ``inf * (xt - mu_t) = nan``.
        path = _LinearTestPath(1.0, sampler_cfg=_FakeSamplerCfg())
        t = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
        x0 = torch.randn(2, 4, dtype=torch.float64)
        x1 = torch.randn(2, 4, dtype=torch.float64)
        xt = torch.randn(2, 4, dtype=torch.float64)

        ut = path.compute_ut(t, xt, x0, x1, generator=torch.Generator())

        assert torch.isfinite(ut).all()
        # coeff is guarded to 0 => ut collapses to dot_mu_t
        torch.testing.assert_close(ut, -x0 + x1)

    def test_matches_jvp_of_compute_xt(self) -> None:
        """``compute_ut`` is the time-derivative of ``compute_xt`` along a fixed noise draw."""
        x0 = torch.randn(5, 3, dtype=torch.float64)
        x1 = torch.randn(5, 3, dtype=torch.float64)
        eps = torch.randn(5, 3, dtype=torch.float64)
        path = _LinearTestPath(1.0, sampler_cfg=_FixedSamplerCfg(eps))

        def xt_of(t: torch.Tensor) -> torch.Tensor:
            return path.compute_xt(t, torch.zeros_like(x0), x0, x1, generator=torch.Generator())

        t = torch.tensor([[0.1], [0.3], [0.5], [0.7], [0.9]], dtype=torch.float64)
        xt, dxt_dt = torch.func.jvp(xt_of, (t,), (torch.ones_like(t),))

        ut = path.compute_ut(t, xt, x0, x1, generator=torch.Generator())
        torch.testing.assert_close(ut, dxt_dt)


# ---------------------------------------------------------------------------
# Broadcasting
# ---------------------------------------------------------------------------


class TestBroadcasting:
    """``t`` may be scalar, ``(B,1)``, or ``(B,D)`` — paths broadcast to ``x0``."""

    @pytest.mark.parametrize("t_shape", [(), (1,), (2, 1), (2, 4)])
    def test_compute_xt_broadcast(self, t_shape: tuple[int, ...]) -> None:
        path = _LinearTestPath(0.0)
        t = torch.full(t_shape, 0.5)
        x0 = torch.randn(2, 4)
        x1 = torch.randn(2, 4)
        out = path.compute_xt(t, torch.zeros_like(x0), x0, x1, generator=torch.Generator())
        assert out.shape == x0.shape

    @pytest.mark.parametrize("t_shape", [(), (1,), (2, 1), (2, 4)])
    def test_compute_ut_broadcast(self, t_shape: tuple[int, ...]) -> None:
        path = _LinearTestPath(1.0, sampler_cfg=_FakeSamplerCfg())
        t = torch.full(t_shape, 0.5)
        x0 = torch.randn(2, 4)
        x1 = torch.randn(2, 4)
        xt = torch.randn(2, 4)
        out = path.compute_ut(t, xt, x0, x1, generator=torch.Generator())
        assert out.shape == x0.shape
