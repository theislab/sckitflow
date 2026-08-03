"""Contracts every probability path must satisfy.

Written because ``VariancePreservingDiracProbabilityPath`` ran **backwards** (``mu(0) == x1``,
``mu(1) == x0``) while ``OTFMObjective`` binds ``x0=source, x1=target`` and ``integrate_translation``
starts at ``y0=source`` and integrates ``t: 0 -> 1``. The class was internally consistent -- its
``compute_ut`` was the correct derivative of its own reversed ``compute_mu_t`` -- so nothing local looked
wrong, and no test in the suite referenced the class at all. Hence these are cross-path contracts, not
per-class unit tests: orientation is only meaningful relative to the other paths and to the solver.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from sc_flow.flow.probability_paths import _probability_paths as P  # noqa: E402

PATHS = [
    "LinearDiracProbabilityPath",
    "LinearGaussianProbabilityPath",
    "SchrodingerBridgeProbabilityPath",
    "VariancePreservingDiracProbabilityPath",
]


def _make(name: str):
    cls = getattr(P, name)
    if getattr(cls, "_require_prng", False):
        return cls(sigma=0.1, prng=torch.Generator().manual_seed(0))
    return cls(sigma=0.0)


@pytest.mark.parametrize("name", PATHS)
def test_orientation_is_x0_to_x1(name: str) -> None:
    """``mu(0) == x0`` and ``mu(1) == x1`` for EVERY path.

    The objective and the ODE solver both treat ``t=1`` as the target; a path that disagrees transports
    in the wrong direction while every local unit test still passes.
    """
    p = _make(name)
    x0 = torch.zeros(4, 3, dtype=torch.float64)
    x1 = torch.ones(4, 3, dtype=torch.float64)
    mu0 = p.compute_mu_t(torch.zeros(4, 1, dtype=torch.float64), x0, x1)
    mu1 = p.compute_mu_t(torch.ones(4, 1, dtype=torch.float64), x0, x1)
    assert torch.allclose(mu0, x0, atol=1e-12), f"{name}: mu(0) != x0 (orientation is reversed)"
    assert torch.allclose(mu1, x1, atol=1e-12), f"{name}: mu(1) != x1 (orientation is reversed)"


@pytest.mark.parametrize("name", PATHS)
def test_compute_ut_is_the_time_derivative_of_compute_mu_t(name: str) -> None:
    """``u_t`` must be ``d mu_t / dt``, checked by central difference in float64.

    Catches an orientation fix applied to ``mu_t`` but not to ``u_t`` (or vice versa) -- the two must move
    together, and each alone looks self-consistent.
    """
    p = _make(name)
    torch.manual_seed(0)
    x0 = torch.randn(32, 5, dtype=torch.float64)
    x1 = torch.randn(32, 5, dtype=torch.float64)
    t = torch.full((32, 1), 0.37, dtype=torch.float64)
    h = 1e-6
    fd = (p.compute_mu_t(t + h, x0, x1) - p.compute_mu_t(t - h, x0, x1)) / (2 * h)
    # u_t is defined on x_t; for the deterministic paths x_t == mu_t, and for the stochastic ones the
    # noise-dependent drift vanishes when evaluated AT the mean.
    ut = p.compute_ut(t, p.compute_mu_t(t, x0, x1), x0, x1)
    assert torch.allclose(ut, fd, atol=1e-5), f"{name}: u_t is not d(mu_t)/dt (max {(ut - fd).abs().max()})"


def test_variance_preserving_path_actually_preserves_variance() -> None:
    """The defining property: ``Var(x_t)`` is flat in ``t`` for independent, equal-variance endpoints.

    The linear interpolant instead dips to half at ``t = 1/2``; this is the whole reason the trig path
    exists, so it is the property worth pinning.
    """
    vp = _make("VariancePreservingDiracProbabilityPath")
    lin = _make("LinearDiracProbabilityPath")
    torch.manual_seed(0)
    x0 = torch.randn(200_000, 1, dtype=torch.float64)
    x1 = torch.randn(200_000, 1, dtype=torch.float64)
    vp_var, lin_var = [], []
    for tv in (0.0, 0.25, 0.5, 0.75, 1.0):
        t = torch.full((200_000, 1), tv, dtype=torch.float64)
        vp_var.append(vp.compute_mu_t(t, x0, x1).var().item())
        lin_var.append(lin.compute_mu_t(t, x0, x1).var().item())
    assert max(vp_var) - min(vp_var) < 0.02, f"VP variance is not flat: {vp_var}"
    assert min(lin_var) < 0.6, f"linear interpolant should dip at t=1/2, got {lin_var}"


@pytest.mark.parametrize("name", PATHS)
def test_endpoint_reparam_is_exact_only_for_linear_dirac(name: str) -> None:
    """``x1 == x_t + (1-t) u_t`` is exact for the straight, noiseless path and not otherwise.

    A downstream gene-space loss uses this reparam, so which paths admit it must be pinned rather than
    remembered: LinearDirac exact; LinearGaussian/SchrodingerBridge unbiased with a sigma-scale spread;
    VariancePreserving biased because the path is curved.
    """
    p = _make(name)
    torch.manual_seed(0)
    x0 = torch.randn(4096, 4, dtype=torch.float64)
    x1 = torch.randn(4096, 4, dtype=torch.float64)
    t = torch.rand(4096, 1, dtype=torch.float64) * 0.8 + 0.1
    xt = p.compute_xt(t, x0, x1)
    err = (xt + (1 - t) * p.compute_ut(t, xt, x0, x1)) - x1
    if name == "LinearDiracProbabilityPath":
        assert err.abs().max() < 1e-10
    elif name == "VariancePreservingDiracProbabilityPath":
        # Curved path: the linear extrapolation is systematically wrong. Threshold is deliberately loose
        # (measured std 0.45 after the orientation fix; it was 1.53 while the path ran backwards, so a
        # tight bound here would encode the bug rather than the contract).
        assert err.std() > 0.1, f"curved path should NOT admit the linear reparam (std {err.std():.3f})"
    else:
        assert abs(err.mean()) < 0.02, f"{name}: reparam should be unbiased, mean {err.mean()}"
