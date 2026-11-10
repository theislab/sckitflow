"""Tests for the concrete probability paths."""

import pytest
import torch

from sckitflow import IsotropicGaussianNoiseSamplerConfig
from sckitflow._constants import PI
from sckitflow.core.probability_paths._base import BasePath, OneSidedPath
from sckitflow.core.probability_paths._probability_paths import (
    EncDecTwoSidedPath,
    LinearOneSidedPath,
    LinearTwoSidedPath,
    TrigOneSidedPath,
    TrigTwoSidedPath,
)

TWO_SIDED = [LinearTwoSidedPath, TrigTwoSidedPath, EncDecTwoSidedPath]
ONE_SIDED = [LinearOneSidedPath, TrigOneSidedPath]
ALL = TWO_SIDED + ONE_SIDED


_GAUSS = IsotropicGaussianNoiseSamplerConfig()


def _path(cls, sigma: float = 1.0) -> BasePath:
    """One-sided paths take no sampler; two-sided ones need it for ``sigma > 0``."""
    return cls(sigma) if issubclass(cls, OneSidedPath) else cls(sigma, sampler_cfg=_GAUSS)


# ---------------------------------------------------------------------------
# Instantiation / MRO
# ---------------------------------------------------------------------------


class TestInstantiation:
    @pytest.mark.parametrize("cls", ALL)
    def test_all_concrete(self, cls) -> None:
        p = _path(cls)
        assert isinstance(p, BasePath)
        assert p.sigma == 1.0

    @pytest.mark.parametrize("cls", TWO_SIDED)
    def test_zero_sigma_no_sampler(self, cls) -> None:
        p = cls(0.0)
        assert p.sigma == 0.0

    @pytest.mark.parametrize("sigma", [0.0, 0.5, 2.0])
    @pytest.mark.parametrize("cls", ONE_SIDED)
    def test_one_sided_requires_unit_sigma(self, cls, sigma) -> None:
        # Inference starts from the unscaled latent, so the path must start from x0 itself.
        with pytest.raises(ValueError, match="needs sigma == 1"):
            cls(sigma)

    @pytest.mark.parametrize("cls", ONE_SIDED)
    def test_one_sided_rejects_sampler(self, cls) -> None:
        with pytest.raises(ValueError, match="takes no `sampler_cfg`"):
            cls(1.0, sampler_cfg=_GAUSS)

    def test_mixin_mro_linear(self) -> None:
        # _LinearPathMixin provides compute_beta_t = t.
        t = torch.linspace(0, 1, 11)
        for cls in (LinearTwoSidedPath, LinearOneSidedPath):
            p = _path(cls)
            torch.testing.assert_close(p.compute_beta_t(t), t)

    def test_mixin_mro_trig(self) -> None:
        # _TrigPathMixin provides compute_beta_t = sin(pi t / 2).
        t = torch.linspace(0, 1, 11)
        expected = torch.sin(0.5 * PI * t)
        for cls in (TrigTwoSidedPath, TrigOneSidedPath):
            p = _path(cls)
            torch.testing.assert_close(p.compute_beta_t(t), expected)

    def test_one_sided_alpha_is_zero(self) -> None:
        t = torch.linspace(0, 1, 11)
        for cls in ONE_SIDED:
            p = _path(cls)
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

    @pytest.mark.parametrize("cls", [LinearTwoSidedPath, TrigTwoSidedPath])
    def test_sigma_scales_std(self, cls) -> None:
        p = cls(0.3, sampler_cfg=_GAUSS)
        t = torch.linspace(0.05, 0.95, 10, dtype=torch.float64)
        torch.testing.assert_close(p.compute_sigma_t(t), 0.3 * torch.sqrt(t * (1 - t)))

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
        p = LinearOneSidedPath(1.0)
        t = torch.linspace(0, 1, 11)
        torch.testing.assert_close(p.compute_alpha_t(t), torch.zeros_like(t))
        torch.testing.assert_close(p.compute_beta_t(t), t)
        torch.testing.assert_close(p.compute_sigma_t(t), 1 - t)


class TestTrigOneSidedPath:
    def test_coefficients(self) -> None:
        p = TrigOneSidedPath(1.0)
        t = torch.linspace(0, 1, 11)
        torch.testing.assert_close(p.compute_alpha_t(t), torch.zeros_like(t))
        torch.testing.assert_close(p.compute_beta_t(t), torch.sin(0.5 * PI * t))
        torch.testing.assert_close(p.compute_sigma_t(t), torch.cos(0.5 * PI * t))


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
        p = _path(cls)
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
        torch.testing.assert_close(p.compute_xt(t0, x0, x1, generator=torch.Generator()), x0)
        torch.testing.assert_close(p.compute_xt(t1, x0, x1, generator=torch.Generator()), x1)

    @pytest.mark.parametrize("cls", ONE_SIDED)
    def test_one_sided_anchors_at_t1(self, cls) -> None:
        p = cls(1.0)
        x0 = torch.randn(3, 4, dtype=torch.float64)
        x1 = torch.randn(3, 4, dtype=torch.float64)
        t1 = torch.ones(3, 1, dtype=torch.float64)
        # beta(1) = 1, sigma(1) = 0  =>  xt(1) = x1 exactly.
        torch.testing.assert_close(p.compute_xt(t1, x0, x1, generator=torch.Generator()), x1)

    @pytest.mark.parametrize("cls", ONE_SIDED)
    def test_one_sided_uses_x0_as_noise(self, cls) -> None:
        # x0 is the noise: x_t = beta_t x1 + sigma_t x0, starting at x0, with no draw of its own.
        p = cls(1.0)
        x0 = torch.randn(3, 4, dtype=torch.float64)
        x1 = torch.randn(3, 4, dtype=torch.float64)
        t0 = torch.zeros(3, 1, dtype=torch.float64)
        t = torch.full((3, 1), 0.3, dtype=torch.float64)
        torch.testing.assert_close(p.compute_xt(t0, x0, x1), x0)
        expected = p.compute_beta_t(t) * x1 + p.compute_sigma_t(t) * x0
        torch.testing.assert_close(p.compute_xt(t, x0, x1, generator=torch.Generator().manual_seed(0)), expected)
        torch.testing.assert_close(p.compute_xt(t, x0, x1, generator=torch.Generator().manual_seed(1)), expected)

    @pytest.mark.parametrize("cls", ALL)
    def test_shape_and_dtype(self, cls) -> None:
        p = _path(cls)
        x0 = torch.randn(2, 3, dtype=torch.float64)
        x1 = torch.randn(2, 3, dtype=torch.float64)
        t = torch.full((2, 1), 0.5, dtype=torch.float64)
        xt = p.compute_xt(t, x0, x1, generator=torch.Generator())
        assert xt.shape == x0.shape
        assert xt.dtype == x0.dtype

    @pytest.mark.parametrize("cls", TWO_SIDED)
    def test_deterministic_with_sigma_zero(self, cls) -> None:
        p = cls(0.0)
        x0 = torch.randn(2, 3)
        x1 = torch.randn(2, 3)
        t = torch.full((2, 1), 0.5)
        a = p.compute_xt(t, x0, x1, generator=torch.Generator().manual_seed(0))
        b = p.compute_xt(t, x0, x1, generator=torch.Generator().manual_seed(1))
        torch.testing.assert_close(a, b)


# ---------------------------------------------------------------------------
# Generator-only determinism
# ---------------------------------------------------------------------------


class TestGeneratorDeterminism:
    @pytest.mark.parametrize("cls", TWO_SIDED)
    def test_same_seed_same_result(self, cls) -> None:
        p = cls(1.0, sampler_cfg=_GAUSS)
        x0 = torch.zeros(4, 3)
        x1 = torch.ones(4, 3)
        t = torch.full((4, 1), 0.5)
        a = p.compute_xt(t, x0, x1, generator=torch.Generator().manual_seed(0))
        b = p.compute_xt(t, x0, x1, generator=torch.Generator().manual_seed(0))
        assert torch.equal(a, b)

    @pytest.mark.parametrize("cls", TWO_SIDED)
    def test_different_seed_different_result(self, cls) -> None:
        p = cls(1.0, sampler_cfg=_GAUSS)
        x0 = torch.zeros(4, 3)
        x1 = torch.ones(4, 3)
        t = torch.full((4, 1), 0.5)
        a = p.compute_xt(t, x0, x1, generator=torch.Generator().manual_seed(0))
        b = p.compute_xt(t, x0, x1, generator=torch.Generator().manual_seed(1))
        assert not torch.equal(a, b)

    @pytest.mark.parametrize("cls", TWO_SIDED)
    def test_generator_state_advances(self, cls) -> None:
        # Calling compute_xt twice with the same generator must consume
        # entropy — otherwise the sampler isn't actually being called.
        p = cls(1.0, sampler_cfg=_GAUSS)
        x0 = torch.zeros(4, 3)
        x1 = torch.ones(4, 3)
        t = torch.full((4, 1), 0.5)
        g = torch.Generator().manual_seed(0)
        a = p.compute_xt(t, x0, x1, generator=g)
        b = p.compute_xt(t, x0, x1, generator=g)
        assert not torch.equal(a, b)


# ---------------------------------------------------------------------------
# compute_ut: JVP cross-check + endpoint robustness
# ---------------------------------------------------------------------------


class TestComputeUt:
    @pytest.mark.parametrize("cls", ALL)
    def test_velocity_matches_jvp(self, cls) -> None:
        """``compute_ut`` should equal ``d/dt compute_xt`` under fixed noise."""
        x0 = torch.randn(5, 3, dtype=torch.float64)
        x1 = torch.randn(5, 3, dtype=torch.float64)
        p = _path(cls, 1.0 if cls in ONE_SIDED else 0.7)

        def xt_of(t: torch.Tensor) -> torch.Tensor:  # a fixed seed fixes the noise draw
            return p.compute_xt(t, x0, x1, generator=torch.Generator().manual_seed(0))

        t = torch.tensor([[0.1], [0.3], [0.5], [0.7], [0.9]], dtype=torch.float64)
        xt, dxt_dt = torch.func.jvp(xt_of, (t,), (torch.ones_like(t),))
        ut = p.compute_ut(t, xt, x0, x1)
        torch.testing.assert_close(ut, dxt_dt)

    @pytest.mark.parametrize("cls", ALL)
    def test_velocity_finite_at_endpoints(self, cls) -> None:
        # The base eps guard must keep ut finite even where sigma_t = 0.
        p = _path(cls)
        x0 = torch.randn(2, 3, dtype=torch.float64)
        x1 = torch.randn(2, 3, dtype=torch.float64)
        xt = torch.randn(2, 3, dtype=torch.float64)
        for t_val in (0.0, 1.0):
            t = torch.full((2, 1), t_val, dtype=torch.float64)
            ut = p.compute_ut(t, xt, x0, x1)
            assert torch.isfinite(ut).all(), f"{cls.__name__} at t={t_val}"


# ---------------------------------------------------------------------------
# Broadcasting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", ALL)
@pytest.mark.parametrize("t_shape", [(), (1,), (2, 1), (2, 4)])
def test_t_broadcast(cls, t_shape) -> None:
    p = _path(cls)
    x0 = torch.randn(2, 4)
    x1 = torch.randn(2, 4)
    xt_in = torch.randn(2, 4)
    t = torch.full(t_shape, 0.5)
    xt = p.compute_xt(t, x0, x1, generator=torch.Generator())
    ut = p.compute_ut(t, xt_in, x0, x1)
    assert xt.shape == x0.shape
    assert ut.shape == x0.shape
