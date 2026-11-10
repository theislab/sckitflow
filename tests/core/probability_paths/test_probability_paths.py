"""Tests for the concrete probability paths."""

import math

import pytest
import torch

from sckitflow import IsotropicGaussianNoiseSamplerConfig
from sckitflow.core.probability_paths._base import BasePath
from sckitflow.core.probability_paths._probability_paths import EncDecPath, LinearPath, TrigPath

ALL = [LinearPath, TrigPath, EncDecPath]

_GAUSS = IsotropicGaussianNoiseSamplerConfig()
_R = math.sqrt(0.5)


def _path(cls, sigma: float = 1.0) -> BasePath:
    return cls(sigma, sampler_cfg=_GAUSS)


# alpha_t, beta_t and sigma_t at t = 0, 1/4, 1/2, 3/4, 1, for sigma = 2.
# Trig at 1/4 and 3/4 is covered by test_trig_preserves_unit_norm.
_COEFFS = {
    LinearPath: (
        [1.0, 0.75, 0.5, 0.25, 0.0],
        [0.0, 0.25, 0.5, 0.75, 1.0],
        [0.0, math.sqrt(3) / 2, 1.0, math.sqrt(3) / 2, 0.0],
    ),
    TrigPath: (
        [1.0, None, _R, None, 0.0],
        [0.0, None, _R, None, 1.0],
        [0.0, math.sqrt(3) / 2, 1.0, math.sqrt(3) / 2, 0.0],
    ),
    EncDecPath: (
        [1.0, 0.5, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.5, 1.0],
        [0.0, 1.0, 2.0, 1.0, 0.0],
    ),
}


@pytest.mark.parametrize("cls", ALL)
def test_coefficients(cls) -> None:
    p = _path(cls, 2.0)
    t = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float64)
    for fn, expected in zip((p.compute_alpha_t, p.compute_beta_t, p.compute_sigma_t), _COEFFS[cls], strict=True):
        known = [i for i, v in enumerate(expected) if v is not None]
        torch.testing.assert_close(fn(t)[known], torch.tensor([expected[i] for i in known], dtype=torch.float64))


def test_trig_preserves_unit_norm() -> None:
    p = _path(TrigPath)
    t = torch.linspace(0, 1, 11, dtype=torch.float64)
    torch.testing.assert_close(p.compute_alpha_t(t) ** 2 + p.compute_beta_t(t) ** 2, torch.ones_like(t))


def test_enc_dec_rejects_zero_sigma() -> None:
    with pytest.raises(ValueError, match="sigma > 0"):
        EncDecPath(0.0)


# dot alpha_t and dot beta_t at given t.
_DOTS = [
    (LinearPath, [0.2, 0.9], [-1.0, -1.0], [1.0, 1.0]),
    (TrigPath, [0.0, 1.0], [0.0, -math.pi / 2], [math.pi / 2, 0.0]),
    (EncDecPath, [0.25, 0.75], [-math.pi, 0.0], [0.0, math.pi]),
]


@pytest.mark.parametrize(("cls", "t", "dot_alpha", "dot_beta"), _DOTS, ids=[c.__name__ for c, *_ in _DOTS])
def test_derivatives(cls, t, dot_alpha, dot_beta) -> None:
    p = _path(cls)
    t = torch.tensor(t, dtype=torch.float64)
    torch.testing.assert_close(p.compute_alpha_t_and_dot_alpha_t(t)[1], torch.tensor(dot_alpha, dtype=torch.float64))
    torch.testing.assert_close(p.compute_beta_t_and_dot_beta_t(t)[1], torch.tensor(dot_beta, dtype=torch.float64))


@pytest.mark.parametrize("cls", ALL)
def test_anchors_at_data(cls) -> None:
    p = _path(cls)
    x0 = torch.randn(3, 4, dtype=torch.float64)
    x1 = torch.randn(3, 4, dtype=torch.float64)
    t0 = torch.zeros(3, 1, dtype=torch.float64)
    t1 = torch.ones(3, 1, dtype=torch.float64)
    # sigma_t(0) = sigma_t(1) = 0 for every path, so the output is exactly x0 / x1 whatever the noise.
    torch.testing.assert_close(p.compute_xt(t0, x0, x1, generator=torch.Generator()), x0)
    torch.testing.assert_close(p.compute_xt(t1, x0, x1, generator=torch.Generator()), x1)


@pytest.mark.parametrize("cls", ALL)
def test_velocity_matches_jvp(cls) -> None:
    """``compute_ut`` should equal ``d/dt compute_xt`` under fixed noise."""
    x0 = torch.randn(5, 3, dtype=torch.float64)
    x1 = torch.randn(5, 3, dtype=torch.float64)
    p = _path(cls, 0.7)

    def xt_of(t: torch.Tensor) -> torch.Tensor:  # a fixed seed fixes the noise draw
        return p.compute_xt(t, x0, x1, generator=torch.Generator().manual_seed(0))

    t = torch.tensor([[0.1], [0.3], [0.5], [0.7], [0.9]], dtype=torch.float64)
    xt, dxt_dt = torch.func.jvp(xt_of, (t,), (torch.ones_like(t),))
    torch.testing.assert_close(p.compute_ut(t, xt, x0, x1), dxt_dt)


@pytest.mark.parametrize("cls", ALL)
def test_velocity_finite_at_endpoints(cls) -> None:
    # The eps guard in `BasePath.compute_ut` must keep ut finite where sigma_t = 0.
    p = _path(cls)
    x0 = torch.randn(2, 3, dtype=torch.float64)
    x1 = torch.randn(2, 3, dtype=torch.float64)
    xt = torch.randn(2, 3, dtype=torch.float64)
    t = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
    assert torch.isfinite(p.compute_ut(t, xt, x0, x1)).all()


@pytest.mark.parametrize("cls", ALL)
@pytest.mark.parametrize(
    "t",
    [
        torch.tensor(0.4),
        torch.tensor([0.4]),
        torch.tensor([0.3, 0.6]),  # (B,), as CFM passes it
        torch.tensor([[0.3], [0.6]]),
        torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]),
    ],
    ids=["()", "(1,)", "(B,)", "(B,1)", "(B,D)"],
)
def test_t_broadcast(cls, t) -> None:
    """Every t shape gives the same xt and ut as t expanded to the data's shape."""
    p = _path(cls)
    x0 = torch.randn(2, 4, dtype=torch.float64)
    x1 = torch.randn(2, 4, dtype=torch.float64)
    xt_in = torch.randn(2, 4, dtype=torch.float64)
    t = t.double()
    t_full = (t[:, None] if t.shape == (2,) else t).expand(2, 4)

    def xt_of(t):
        return p.compute_xt(t, x0, x1, generator=torch.Generator().manual_seed(0))

    torch.testing.assert_close(xt_of(t), xt_of(t_full))
    torch.testing.assert_close(p.compute_ut(t, xt_in, x0, x1), p.compute_ut(t_full, xt_in, x0, x1))
