import pytest
import torch

from sckitflow.core.probability_paths import (
    EncDecTwoSidedPath,
    LinearOneSidedPath,
    LinearTwoSidedPath,
    OneSidedPath,
    TrigOneSidedPath,
    TrigTwoSidedPath,
    VPOneSidedPath,
)
from sckitflow.core.solvers import ODESolver
from sckitflow.core.surrogate import LossGuidedVelocity, PathFlowMap, guidance_coefficients

PATHS = [LinearOneSidedPath, TrigOneSidedPath, VPOneSidedPath, LinearTwoSidedPath, TrigTwoSidedPath, EncDecTwoSidedPath]


class _Field(torch.nn.Module):
    """A velocity module wrapping a plain ``v(t, x)``."""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def get_vf_fn(self, condition_dict=None, source=None):
        return self.fn


@pytest.mark.parametrize("path_cls", PATHS)
@pytest.mark.parametrize("t", [0.1, 0.3, 0.7, 0.9])
def test_estimator_recovers_x1(path_cls, t):
    path, t = path_cls(sigma=0.0), torch.tensor(t, dtype=torch.float64)
    x0, x1 = torch.randn(2, 8, 3, dtype=torch.float64)
    a_fn = (
        path.compute_sigma_t_and_dot_sigma_t if isinstance(path, OneSidedPath) else path.compute_alpha_t_and_dot_alpha_t
    )
    (a, da), (b, db) = a_fn(t), path.compute_beta_t_and_dot_beta_t(t)
    xt, ut = a * x0 + b * x1, da * x0 + db * x1
    if not isinstance(path, OneSidedPath):  # ties the algebra to the path's own sampler
        torch.testing.assert_close(xt, path.compute_xt(t, xt, x0, x1, generator=None))
        torch.testing.assert_close(ut, path.compute_ut(t, xt, x0, x1, generator=None))
    c_x, c_v, lam = guidance_coefficients(path, t)
    if da * b - a * db == 0:  # EncDec: the 2x2 system for (x0, x1) is singular, guidance off
        assert lam == 0
    else:
        torch.testing.assert_close(c_x * xt + c_v * ut, x1)
        assert lam >= 0


def test_closed_forms():
    t = torch.linspace(0.05, 0.95, 7, dtype=torch.float64)
    c_x, c_v, lam = guidance_coefficients(LinearOneSidedPath(sigma=0.0), t)
    torch.testing.assert_close((c_x, c_v, lam), (torch.ones_like(t), 1 - t, (1 - t) / t))
    c_x, c_v, _ = guidance_coefficients(VPOneSidedPath(sigma=0.0), t)
    torch.testing.assert_close((c_x, c_v), (t, 1 - t**2))


@pytest.mark.parametrize("path_cls", PATHS)
@pytest.mark.parametrize("t", [0.0, 1.0])
def test_endpoints_finite(path_cls, t):
    field = _Field(lambda t, x: -x)
    path = path_cls(sigma=0.0)
    v = LossGuidedVelocity(field, path, lambda x: (x**2).sum(-1), flow_map=PathFlowMap(field, path))
    assert torch.isfinite(v.get_vf_fn()(torch.tensor(t), torch.randn(4, 3))).all()


def test_gradient_matches_analytic():
    # v = A x + c is linear, so x1_hat = M x + (1 - t) c with M = I + (1 - t) A and the guidance is
    # -b_t M^T grad Psi(x1_hat) for Psi(x) = |x - y|^2 / 2.
    torch.manual_seed(0)
    A, c, y = (
        torch.randn(3, 3, dtype=torch.float64),
        torch.randn(3, dtype=torch.float64),
        torch.randn(3, dtype=torch.float64),
    )
    t, x = torch.tensor(0.4, dtype=torch.float64), torch.randn(5, 3, dtype=torch.float64)
    field = _Field(lambda t, x: x @ A.T + c)
    guided = LossGuidedVelocity(
        field,
        LinearOneSidedPath(sigma=0.0),
        lambda x: 0.5 * ((x - y) ** 2).sum(-1),
        flow_map=PathFlowMap(field, LinearOneSidedPath(sigma=0.0)),
        scale=0.5,
    )
    M = torch.eye(3, dtype=torch.float64) + (1 - t) * A
    x1_hat = x @ M.T + (1 - t) * c
    expected = x @ A.T + c - 0.5 * (1 - t) / t * (x1_hat - y) @ M
    torch.testing.assert_close(guided.get_vf_fn()(t, x), expected)


@pytest.mark.parametrize("path_cls", [p for p in PATHS if p is not EncDecTwoSidedPath])
@pytest.mark.parametrize("n_steps", [1, 3])
def test_path_flow_map_follows_curved_paths(path_cls, n_steps):
    # with the exact velocity of a single pair, x_u = a_u x0 + b_u x1, the map lands on x1 from any t,
    # however curved the schedule; plain Euler steps would not on the trig and VP paths
    path, t = path_cls(sigma=0.0), torch.tensor(0.3, dtype=torch.float64)
    x0, x1 = torch.randn(2, 8, 3, dtype=torch.float64)

    def v(u, x):
        (_, da), (_, db) = (
            (
                path.compute_sigma_t_and_dot_sigma_t
                if isinstance(path, OneSidedPath)
                else path.compute_alpha_t_and_dot_alpha_t
            )(u),
            path.compute_beta_t_and_dot_beta_t(u),
        )
        return da * x0 + db * x1

    a_fn = (
        path.compute_sigma_t_and_dot_sigma_t if isinstance(path, OneSidedPath) else path.compute_alpha_t_and_dot_alpha_t
    )
    xt = a_fn(t)[0] * x0 + path.compute_beta_t_and_dot_beta_t(t)[0] * x1
    phi = PathFlowMap(_Field(v), path, n_steps=n_steps).get_map_fn()
    torch.testing.assert_close(phi(t, torch.ones_like(t), xt), x1)


def _gaussian_prior(m, s):
    """Exact marginal velocity of the linear one-sided path for data ``N(m, s^2 I)``."""

    def v(t, x):
        x1_hat = m + t * s**2 / (t**2 * s**2 + (1 - t) ** 2) * (x - t * m)
        return (x1_hat - x) / (1 - t)

    return _Field(v)


def _sample(field, n=4000, steps=100):
    x0 = torch.randn(n, 2, generator=torch.Generator().manual_seed(0))
    return ODESolver(field).solve(x0, torch.linspace(0, 1 - 1e-3, steps))


def test_scale_zero_is_prior():
    prior = _gaussian_prior(torch.tensor([2.0, -1.0]), 0.5)
    guided = LossGuidedVelocity(
        prior,
        LinearOneSidedPath(sigma=0.0),
        lambda x: pytest.fail("evaluated"),
        flow_map=PathFlowMap(prior, LinearOneSidedPath(sigma=0.0)),
        scale=0.0,
    )
    torch.testing.assert_close(_sample(guided), _sample(prior))


def test_guidance_descends_toward_posterior():
    # prior N(m, s^2), Psi = |x - y|^2 / 2 tilts it to N((m + s^2 y) / (1 + s^2), s^2 / (1 + s^2))
    m, y, s = torch.tensor([2.0, -1.0]), torch.tensor([-1.0, 1.0]), 0.5
    prior = _gaussian_prior(m, s)
    potential = lambda x: 0.5 * ((x - y) ** 2).sum(-1)
    unguided = _sample(prior)
    guided = _sample(
        LossGuidedVelocity(
            prior, LinearOneSidedPath(sigma=0.0), potential, flow_map=PathFlowMap(prior, LinearOneSidedPath(sigma=0.0))
        )
    )
    post_mean = (m + s**2 * y) / (1 + s**2)
    assert potential(guided).mean() < potential(unguided).mean()
    torch.testing.assert_close(unguided.mean(0), m, atol=0.05, rtol=0)
    torch.testing.assert_close(guided.mean(0), post_mean, atol=0.1, rtol=0)


def test_raises_under_inference_mode():
    field = _Field(lambda t, x: -x)
    guided = LossGuidedVelocity(
        field,
        LinearOneSidedPath(sigma=0.0),
        lambda x: x.sum(-1),
        flow_map=PathFlowMap(field, LinearOneSidedPath(sigma=0.0)),
    )
    with torch.inference_mode(), pytest.raises(RuntimeError, match="autograd"):
        guided.get_vf_fn()(torch.tensor(0.5), torch.randn(2, 3))
