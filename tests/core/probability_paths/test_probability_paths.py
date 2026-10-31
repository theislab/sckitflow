"""Tests for the concrete probability paths."""

import pytest
import torch

from sckitflow._constants import PI
from sckitflow.core.probability_paths._base import BasePath
from sckitflow.core.probability_paths._probability_paths import (
    EncDecTwoSidedPath,
    LinearOneSidedPath,
    LinearTwoSidedPath,
    TrigOneSidedPath,
    TrigTwoSidedPath,
    VPOneSidedPath,
)

TWO_SIDED = [LinearTwoSidedPath, TrigTwoSidedPath, EncDecTwoSidedPath]
ONE_SIDED = [LinearOneSidedPath, TrigOneSidedPath, VPOneSidedPath]
ALL = TWO_SIDED + ONE_SIDED


# ---------------------------------------------------------------------------
# Test sampler configs
# ---------------------------------------------------------------------------


class _GaussianSamplerConfig:
    """Minimal std-Gaussian sampler config."""

    def build(self):
        def sampler(shape, *, generator, device, dtype):
            return torch.randn(shape, generator=generator, device=device, dtype=dtype)

        return sampler


class _FixedSamplerConfig:
    """Sampler that always returns the same epsilon — for JVP tests."""

    def __init__(self, eps: torch.Tensor) -> None:
        self.eps = eps

    def build(self):
        eps = self.eps

        def sampler(shape, *, generator, device, dtype):
            return eps.to(device=device, dtype=dtype)

        return sampler


_GAUSS = _GaussianSamplerConfig()


# ---------------------------------------------------------------------------
# Instantiation / MRO
# ---------------------------------------------------------------------------


class TestInstantiation:
    @pytest.mark.parametrize("cls", ALL)
    def test_all_concrete(self, cls) -> None:
        p = cls(1.0, sampler_cfg=_GAUSS)
        assert isinstance(p, BasePath)
        assert p.sigma == 1.0

    @pytest.mark.parametrize("cls", ALL)
    def test_zero_sigma_no_sampler(self, cls) -> None:
        p = cls(0.0)
        assert p.sigma == 0.0

    def test_mixin_mro_linear(self) -> None:
        # _LinearPathMixin provides compute_beta_t = t.
        t = torch.linspace(0, 1, 11)
        for cls in (LinearTwoSidedPath, LinearOneSidedPath, VPOneSidedPath):
            p = cls(1.0, sampler_cfg=_GAUSS)
            torch.testing.assert_close(p.compute_beta_t(t), t)

    def test_mixin_mro_trig(self) -> None:
        # _TrigPathMixin provides compute_beta_t = sin(pi t / 2).
        t = torch.linspace(0, 1, 11)
        expected = torch.sin(0.5 * PI * t)
        for cls in (TrigTwoSidedPath, TrigOneSidedPath):
            p = cls(1.0, sampler_cfg=_GAUSS)
            torch.testing.assert_close(p.compute_beta_t(t), expected)

    def test_one_sided_alpha_is_zero(self) -> None:
        t = torch.linspace(0, 1, 11)
        for cls in ONE_SIDED:
            p = cls(1.0, sampler_cfg=_GAUSS)
            torch.testing.assert_close(p.compute_alpha_t(t), torch.zeros_like(t))


# ---------------------------------------------------------------------------
# Coefficient analytic values
# ---------------------------------------------------------------------------


class TestLinearTwoSidedPath:
    def test_alpha_beta(self) -> None:
        p = LinearTwoSidedPath(1.0, sampler_cfg=_GAUSS)
        t = torch.linspace(0, 1, 11)
        torch.testing.assert_close(p.compute_alpha_t(t), 1 - t)
        torch.testing.assert_close(p.compute_beta_t(t), t)

    def test_sigma(self) -> None:
        p = LinearTwoSidedPath(1.0, sampler_cfg=_GAUSS)
        t = torch.linspace(0.05, 0.95, 10, dtype=torch.float64)
        torch.testing.assert_close(p.compute_sigma_t(t), torch.sqrt(t * (1 - t)))

    def test_sigma_scales_sqrt_sigma(self) -> None:
        p1 = LinearTwoSidedPath(1.0, sampler_cfg=_GAUSS)
        p4 = LinearTwoSidedPath(4.0, sampler_cfg=_GAUSS)
        t = torch.tensor([0.5], dtype=torch.float64)
        torch.testing.assert_close(p4.compute_sigma_t(t), 2 * p1.compute_sigma_t(t))

    def test_sigma_zero_at_endpoints(self) -> None:
        p = LinearTwoSidedPath(1.0, sampler_cfg=_GAUSS)
        t = torch.tensor([0.0, 1.0])
        torch.testing.assert_close(p.compute_sigma_t(t), torch.zeros_like(t))


class TestTrigTwoSidedPath:
    def test_alpha_beta(self) -> None:
        p = TrigTwoSidedPath(1.0, sampler_cfg=_GAUSS)
        t = torch.linspace(0, 1, 11)
        torch.testing.assert_close(p.compute_alpha_t(t), torch.cos(0.5 * PI * t))
        torch.testing.assert_close(p.compute_beta_t(t), torch.sin(0.5 * PI * t))

    def test_sigma(self) -> None:
        p = TrigTwoSidedPath(1.0, sampler_cfg=_GAUSS)
        t = torch.linspace(0.05, 0.95, 10, dtype=torch.float64)
        torch.testing.assert_close(p.compute_sigma_t(t), torch.sqrt(t * (1 - t)))

    def test_sigma_zero_at_endpoints(self) -> None:
        p = TrigTwoSidedPath(1.0, sampler_cfg=_GAUSS)
        t = torch.tensor([0.0, 1.0])
        torch.testing.assert_close(p.compute_sigma_t(t), torch.zeros_like(t))


class TestEncDecTwoSidedPath:
    def test_alpha_piecewise(self) -> None:
        p = EncDecTwoSidedPath(1.0, sampler_cfg=_GAUSS)
        # t <= 0.5: alpha active, beta = 0
        t_left = torch.tensor([0.0, 0.1, 0.25, 0.5], dtype=torch.float64)
        torch.testing.assert_close(p.compute_alpha_t(t_left), torch.cos(PI * t_left) ** 2)
        torch.testing.assert_close(p.compute_beta_t(t_left), torch.zeros_like(t_left))
        # t > 0.5: alpha = 0, beta active
        t_right = torch.tensor([0.75, 0.9, 1.0], dtype=torch.float64)
        torch.testing.assert_close(p.compute_alpha_t(t_right), torch.zeros_like(t_right))
        torch.testing.assert_close(p.compute_beta_t(t_right), torch.cos(PI * t_right) ** 2)

    def test_sigma(self) -> None:
        p = EncDecTwoSidedPath(1.0, sampler_cfg=_GAUSS)
        t = torch.linspace(0, 1, 11)
        torch.testing.assert_close(p.compute_sigma_t(t), torch.sin(PI * t) ** 2)

    def test_sigma_scales_linearly(self) -> None:
        p1 = EncDecTwoSidedPath(1.0, sampler_cfg=_GAUSS)
        p2 = EncDecTwoSidedPath(2.0, sampler_cfg=_GAUSS)
        t = torch.tensor([0.5], dtype=torch.float64)
        torch.testing.assert_close(p2.compute_sigma_t(t), 2 * p1.compute_sigma_t(t))

    def test_sigma_zero_at_endpoints(self) -> None:
        p = EncDecTwoSidedPath(1.0, sampler_cfg=_GAUSS)
        t = torch.tensor([0.0, 1.0])
        torch.testing.assert_close(p.compute_sigma_t(t), torch.zeros_like(t))

    def test_sigma_peak_at_half(self) -> None:
        p = EncDecTwoSidedPath(1.0, sampler_cfg=_GAUSS)
        t = torch.tensor([0.5])
        torch.testing.assert_close(p.compute_sigma_t(t), torch.ones(1))


class TestLinearOneSidedPath:
    def test_coefficients(self) -> None:
        p = LinearOneSidedPath(1.0, sampler_cfg=_GAUSS)
        t = torch.linspace(0, 1, 11)
        torch.testing.assert_close(p.compute_alpha_t(t), torch.zeros_like(t))
        torch.testing.assert_close(p.compute_beta_t(t), t)
        torch.testing.assert_close(p.compute_sigma_t(t), 1 - t)


class TestTrigOneSidedPath:
    def test_coefficients(self) -> None:
        p = TrigOneSidedPath(1.0, sampler_cfg=_GAUSS)
        t = torch.linspace(0, 1, 11)
        torch.testing.assert_close(p.compute_alpha_t(t), torch.zeros_like(t))
        torch.testing.assert_close(p.compute_beta_t(t), torch.sin(0.5 * PI * t))
        torch.testing.assert_close(p.compute_sigma_t(t), torch.cos(0.5 * PI * t))


class TestVPOneSidedPath:
    def test_coefficients(self) -> None:
        p = VPOneSidedPath(1.0, sampler_cfg=_GAUSS)
        t = torch.linspace(0, 1, 11)
        torch.testing.assert_close(p.compute_alpha_t(t), torch.zeros_like(t))
        torch.testing.assert_close(p.compute_beta_t(t), t)
        torch.testing.assert_close(p.compute_sigma_t(t), torch.sqrt(1 - t**2))


# NOTE: one-sided sigma_t does *not* use ``self.sigma`` — only the base
# ``compute_xt``/``compute_ut`` gate on ``sigma > 0``. Document that quirk:
@pytest.mark.parametrize("cls", ONE_SIDED)
def test_one_sided_sigma_ignores_sigma_scalar(cls) -> None:
    p1 = cls(1.0, sampler_cfg=_GAUSS)
    p4 = cls(4.0, sampler_cfg=_GAUSS)
    t = torch.linspace(0, 1, 11)
    torch.testing.assert_close(p1.compute_sigma_t(t), p4.compute_sigma_t(t))


# ---------------------------------------------------------------------------
# VJP derivatives
# ---------------------------------------------------------------------------


class TestVJPDerivatives:
    def test_linear_two_sided(self) -> None:
        p = LinearTwoSidedPath(1.0, sampler_cfg=_GAUSS)
        t = torch.linspace(0.1, 0.9, 5, dtype=torch.float64)
        _, dot_alpha = p.compute_alpha_t_and_dot_alpha_t(t)
        _, dot_beta = p.compute_beta_t_and_dot_beta_t(t)
        torch.testing.assert_close(dot_alpha, -torch.ones_like(t))
        torch.testing.assert_close(dot_beta, torch.ones_like(t))

    def test_trig_two_sided(self) -> None:
        p = TrigTwoSidedPath(1.0, sampler_cfg=_GAUSS)
        t = torch.linspace(0.1, 0.9, 5, dtype=torch.float64)
        _, dot_alpha = p.compute_alpha_t_and_dot_alpha_t(t)
        _, dot_beta = p.compute_beta_t_and_dot_beta_t(t)
        torch.testing.assert_close(dot_alpha, -0.5 * PI * torch.sin(0.5 * PI * t))
        torch.testing.assert_close(dot_beta, 0.5 * PI * torch.cos(0.5 * PI * t))

    def test_enc_dec_piecewise_derivatives(self) -> None:
        p = EncDecTwoSidedPath(1.0, sampler_cfg=_GAUSS)
        t = torch.tensor([0.2, 0.8], dtype=torch.float64)
        _, dot_alpha = p.compute_alpha_t_and_dot_alpha_t(t)
        _, dot_beta = p.compute_beta_t_and_dot_beta_t(t)
        expected_alpha = torch.where(
            t <= 0.5,
            -PI * torch.sin(2 * PI * t),
            torch.zeros_like(t),
        )
        expected_beta = torch.where(
            t > 0.5,
            -PI * torch.sin(2 * PI * t),
            torch.zeros_like(t),
        )
        torch.testing.assert_close(dot_alpha, expected_alpha)
        torch.testing.assert_close(dot_beta, expected_beta)

    @pytest.mark.parametrize("cls", ALL)
    def test_sigma_derivative_finite_interior(self, cls) -> None:
        p = cls(1.0, sampler_cfg=_GAUSS)
        t = torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64)
        sigma, dot_sigma = p.compute_sigma_t_and_dot_sigma_t(t)
        assert torch.isfinite(sigma).all()
        assert torch.isfinite(dot_sigma).all()


# ---------------------------------------------------------------------------
# compute_xt: endpoint semantics
# ---------------------------------------------------------------------------


class TestComputeXtEndpoints:
    @pytest.mark.parametrize("cls", TWO_SIDED)
    def test_two_sided_anchors_at_data(self, cls) -> None:
        p = cls(1.0, sampler_cfg=_GAUSS)
        x0 = torch.randn(3, 4, dtype=torch.float64)
        x1 = torch.randn(3, 4, dtype=torch.float64)
        t0 = torch.zeros(3, 1, dtype=torch.float64)
        t1 = torch.ones(3, 1, dtype=torch.float64)
        # sigma_t(0) = sigma_t(1) = 0 for every two-sided path, so the
        # output is exactly x0 / x1 regardless of the drawn noise.
        torch.testing.assert_close(p.compute_xt(t0, x0, x0, x1, generator=torch.Generator()), x0)
        torch.testing.assert_close(p.compute_xt(t1, x0, x0, x1, generator=torch.Generator()), x1)

    @pytest.mark.parametrize("cls", ONE_SIDED)
    def test_one_sided_anchors_at_t1(self, cls) -> None:
        p = cls(1.0, sampler_cfg=_GAUSS)
        x0 = torch.randn(3, 4, dtype=torch.float64)
        x1 = torch.randn(3, 4, dtype=torch.float64)
        t1 = torch.ones(3, 1, dtype=torch.float64)
        # beta(1) = 1, sigma(1) = 0  =>  xt(1) = x1 exactly.
        torch.testing.assert_close(p.compute_xt(t1, x0, x0, x1, generator=torch.Generator()), x1)

    @pytest.mark.parametrize("cls", ONE_SIDED)
    def test_one_sided_is_noise_at_t0(self, cls) -> None:
        p = cls(1.0, sampler_cfg=_GAUSS)
        x0 = torch.randn(3, 4, dtype=torch.float64)
        x1 = torch.randn(3, 4, dtype=torch.float64)
        t0 = torch.zeros(3, 1, dtype=torch.float64)
        a = p.compute_xt(t0, x0, x0, x1, generator=torch.Generator().manual_seed(0))
        b = p.compute_xt(t0, x0, x0, x1, generator=torch.Generator().manual_seed(1))
        # Pure noise start: generator drives the output, and it is not x0.
        assert not torch.equal(a, b)
        assert not torch.allclose(a, x0)

    @pytest.mark.parametrize("cls", ALL)
    def test_shape_and_dtype(self, cls) -> None:
        p = cls(1.0, sampler_cfg=_GAUSS)
        x0 = torch.randn(2, 3, dtype=torch.float64)
        x1 = torch.randn(2, 3, dtype=torch.float64)
        t = torch.full((2, 1), 0.5, dtype=torch.float64)
        xt = p.compute_xt(t, x0, x0, x1, generator=torch.Generator())
        assert xt.shape == x0.shape
        assert xt.dtype == x0.dtype

    @pytest.mark.parametrize("cls", ALL)
    def test_deterministic_with_sigma_zero(self, cls) -> None:
        p = cls(0.0)
        x0 = torch.randn(2, 3)
        x1 = torch.randn(2, 3)
        t = torch.full((2, 1), 0.5)
        a = p.compute_xt(t, x0, x0, x1, generator=torch.Generator().manual_seed(0))
        b = p.compute_xt(t, x0, x0, x1, generator=torch.Generator().manual_seed(1))
        torch.testing.assert_close(a, b)


# ---------------------------------------------------------------------------
# Generator-only determinism
# ---------------------------------------------------------------------------


class TestGeneratorDeterminism:
    @pytest.mark.parametrize("cls", ALL)
    def test_same_seed_same_result(self, cls) -> None:
        p = cls(1.0, sampler_cfg=_GAUSS)
        x0 = torch.zeros(4, 3)
        x1 = torch.ones(4, 3)
        t = torch.full((4, 1), 0.5)
        a = p.compute_xt(t, x0, x0, x1, generator=torch.Generator().manual_seed(0))
        b = p.compute_xt(t, x0, x0, x1, generator=torch.Generator().manual_seed(0))
        assert torch.equal(a, b)

    @pytest.mark.parametrize("cls", ALL)
    def test_different_seed_different_result(self, cls) -> None:
        p = cls(1.0, sampler_cfg=_GAUSS)
        x0 = torch.zeros(4, 3)
        x1 = torch.ones(4, 3)
        t = torch.full((4, 1), 0.5)
        a = p.compute_xt(t, x0, x0, x1, generator=torch.Generator().manual_seed(0))
        b = p.compute_xt(t, x0, x0, x1, generator=torch.Generator().manual_seed(1))
        assert not torch.equal(a, b)

    @pytest.mark.parametrize("cls", ALL)
    def test_generator_state_advances(self, cls) -> None:
        # Calling compute_xt twice with the same generator must consume
        # entropy — otherwise the sampler isn't actually being called.
        p = cls(1.0, sampler_cfg=_GAUSS)
        x0 = torch.zeros(4, 3)
        x1 = torch.ones(4, 3)
        t = torch.full((4, 1), 0.5)
        g = torch.Generator().manual_seed(0)
        a = p.compute_xt(t, x0, x0, x1, generator=g)
        b = p.compute_xt(t, x0, x0, x1, generator=g)
        assert not torch.equal(a, b)


# ---------------------------------------------------------------------------
# compute_ut: JVP cross-check + endpoint robustness
# ---------------------------------------------------------------------------


class TestComputeUt:
    @pytest.mark.parametrize("cls", ALL)
    def test_velocity_matches_jvp(self, cls) -> None:
        """``compute_ut`` should equal ``d/dt compute_xt`` under fixed noise."""
        x0 = torch.randn(5, 3, dtype=torch.float64)  # was 4
        x1 = torch.randn(5, 3, dtype=torch.float64)  # was 4
        eps = torch.randn(5, 3, dtype=torch.float64)  # was 4
        p = cls(0.7, sampler_cfg=_FixedSamplerConfig(eps))

        def xt_of(t: torch.Tensor) -> torch.Tensor:
            return p.compute_xt(t, x0, x0, x1, generator=torch.Generator())

        t = torch.tensor([[0.1], [0.3], [0.5], [0.7], [0.9]], dtype=torch.float64)
        xt, dxt_dt = torch.func.jvp(xt_of, (t,), (torch.ones_like(t),))
        ut = p.compute_ut(t, xt, x0, x1, generator=torch.Generator())
        torch.testing.assert_close(ut, dxt_dt)

    @pytest.mark.parametrize("cls", ALL)
    def test_velocity_finite_at_endpoints(self, cls) -> None:
        # The base eps guard must keep ut finite even where sigma_t = 0.
        p = cls(1.0, sampler_cfg=_GAUSS)
        x0 = torch.randn(2, 3, dtype=torch.float64)
        x1 = torch.randn(2, 3, dtype=torch.float64)
        xt = torch.randn(2, 3, dtype=torch.float64)
        for t_val in (0.0, 1.0):
            t = torch.full((2, 1), t_val, dtype=torch.float64)
            ut = p.compute_ut(t, xt, x0, x1, generator=torch.Generator())
            assert torch.isfinite(ut).all(), f"{cls.__name__} at t={t_val}"

    @pytest.mark.parametrize("cls", ALL)
    def test_deterministic_ut_with_sigma_zero(self, cls) -> None:
        p = cls(0.0)
        x0 = torch.randn(2, 3, dtype=torch.float64)
        x1 = torch.randn(2, 3, dtype=torch.float64)
        xt = torch.randn(2, 3, dtype=torch.float64)
        t = torch.full((2, 1), 0.5, dtype=torch.float64)
        a = p.compute_ut(t, xt, x0, x1, generator=torch.Generator().manual_seed(0))
        b = p.compute_ut(t, xt, x0, x1, generator=torch.Generator().manual_seed(1))
        torch.testing.assert_close(a, b)


# ---------------------------------------------------------------------------
# Broadcasting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", ALL)
@pytest.mark.parametrize("t_shape", [(), (1,), (2, 1), (2, 4)])
def test_t_broadcast(cls, t_shape) -> None:
    p = cls(1.0, sampler_cfg=_GAUSS)
    x0 = torch.randn(2, 4)
    x1 = torch.randn(2, 4)
    xt_in = torch.randn(2, 4)
    t = torch.full(t_shape, 0.5)
    xt = p.compute_xt(t, x0, x0, x1, generator=torch.Generator())
    ut = p.compute_ut(t, xt_in, x0, x1, generator=torch.Generator())
    assert xt.shape == x0.shape
    assert ut.shape == x0.shape
