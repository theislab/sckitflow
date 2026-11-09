import pytest
import torch

from sckitflow import IsotropicGaussianNoiseSamplerConfig
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
from sckitflow.core.surrogate import (
    LossGuidedVelocity,
    PathFlowMap,
    endpoint_coefficients,
    guidance_strength,
    posterior_coefficients,
)

PATHS = [LinearOneSidedPath, TrigOneSidedPath, VPOneSidedPath, LinearTwoSidedPath, TrigTwoSidedPath, EncDecTwoSidedPath]
SOLVABLE = [p for p in PATHS if p is not EncDecTwoSidedPath]


class _Field(torch.nn.Module):
    """A velocity module wrapping a plain ``v(t, x)``."""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def get_vf_fn(self, condition_dict=None, source=None):
        return self.fn


def _guided(field, potential, path=None, **kwargs):
    path = LinearOneSidedPath(sigma=0.0) if path is None else path
    return LossGuidedVelocity(field, path, potential, flow_map=PathFlowMap(field, path), **kwargs)


@pytest.mark.parametrize("path_cls", SOLVABLE)
@pytest.mark.parametrize("t", [0.1, 0.3, 0.7, 0.9])
def test_posterior_coefficients_recover_the_pair(path_cls, t):
    # draws from the path's own sampler, with noise on the one-sided paths, so the assumed blend is checked
    one_sided = issubclass(path_cls, OneSidedPath)
    path = path_cls(sigma=1.0, sampler_cfg=IsotropicGaussianNoiseSamplerConfig()) if one_sided else path_cls(sigma=0.0)
    t, generator = torch.tensor(t, dtype=torch.float64), torch.Generator().manual_seed(0)
    x0, x1 = torch.randn(2, 8, 3, dtype=torch.float64)
    xt = path.compute_xt(t, None, x0, x1, generator=generator)
    ut = path.compute_ut(t, xt, x0, x1, generator=generator)
    (c0_x, c0_v), (c1_x, c1_v) = posterior_coefficients(path, t)
    torch.testing.assert_close(c1_x * xt + c1_v * ut, x1)
    if not one_sided:  # one-sided paths draw their own x0
        torch.testing.assert_close(c0_x * xt + c0_v * ut, x0)
    assert guidance_strength(path, t) >= 0


def test_closed_forms():
    t = torch.linspace(0.05, 0.95, 7, dtype=torch.float64)
    path = LinearOneSidedPath(sigma=0.0)
    (_, (c_x, c_v)), strength = posterior_coefficients(path, t), guidance_strength(path, t)
    torch.testing.assert_close((c_x, c_v, strength), (torch.ones_like(t), 1 - t, (1 - t) / t))
    _, (c_x, c_v) = posterior_coefficients(VPOneSidedPath(sigma=0.0), t)
    torch.testing.assert_close((c_x, c_v), (t, 1 - t**2))


def test_path_flow_map_rejects_unsolvable_paths():
    with pytest.raises(ValueError, match="does not determine"):
        PathFlowMap(_Field(lambda t, x: -x), EncDecTwoSidedPath(sigma=0.0))


@pytest.mark.parametrize("path_cls", SOLVABLE)
@pytest.mark.parametrize("t", [0.0, 1.0])
def test_endpoints_finite_with_a_singular_potential(path_cls, t):
    # the potential's gradient is NaN everywhere: wherever the strength is 0 none of it may leak in
    potential = lambda x: torch.sqrt(-(x**2).sum(-1))
    velocity = _guided(_Field(lambda t, x: -x), potential, path_cls(sigma=0.0)).get_vf_fn()(
        torch.tensor(t), torch.randn(4, 3)
    )
    if guidance_strength(path_cls(sigma=0.0), torch.tensor(t)) == 0:
        assert torch.isfinite(velocity).all()


def test_zero_strength_skips_the_potential():
    guided = _guided(_Field(lambda t, x: -x), lambda x: pytest.fail("evaluated"))
    torch.testing.assert_close(guided.get_vf_fn()(torch.tensor(1.0), torch.ones(2, 3)), -torch.ones(2, 3))


def test_per_sample_time():
    field, potential = _Field(lambda t, x: -x), lambda x: (x**2).sum(-1)
    t, x = torch.tensor([0.2, 0.5, 0.8, 1.0]), torch.randn(4, 3)
    batched = _guided(field, potential).get_vf_fn()(t, x)
    for i in range(4):
        torch.testing.assert_close(batched[i], _guided(field, potential).get_vf_fn()(t[i], x[i : i + 1])[0])


def test_rejects_a_source():
    with pytest.raises(ValueError, match="noise"):
        _guided(_Field(lambda t, x: -x), lambda x: x.sum(-1)).get_vf_fn(source=torch.zeros(2, 3))


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
    guided = _guided(field, lambda x: 0.5 * ((x - y) ** 2).sum(-1), scale=0.5)
    M = torch.eye(3, dtype=torch.float64) + (1 - t) * A
    x1_hat = x @ M.T + (1 - t) * c
    expected = x @ A.T + c - 0.5 * (1 - t) / t * (x1_hat - y) @ M
    torch.testing.assert_close(guided.get_vf_fn()(t, x), expected)


@pytest.mark.parametrize("path_cls", SOLVABLE)
@pytest.mark.parametrize("n_steps", [1, 3])
def test_path_flow_map_follows_curved_paths(path_cls, n_steps):
    # with the exact velocity of a single pair, x_u = a_u x0 + b_u x1, the map lands on x1 from any t,
    # however curved the schedule; plain Euler steps would not on the trig and VP paths
    path, t = path_cls(sigma=0.0), torch.tensor(0.3, dtype=torch.float64)
    x0, x1 = torch.randn(2, 8, 3, dtype=torch.float64)

    def v(u, x):
        (_, da), (_, db) = endpoint_coefficients(path, u)
        return da * x0 + db * x1

    (a, _), (b, _) = endpoint_coefficients(path, t)
    phi = PathFlowMap(_Field(v), path, n_steps=n_steps).get_map_fn()
    torch.testing.assert_close(phi(t, torch.ones_like(t), a * x0 + b * x1), x1)


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
    guided = _guided(prior, lambda x: pytest.fail("evaluated"), scale=0.0)
    torch.testing.assert_close(_sample(guided), _sample(prior))


def test_guidance_descends_toward_posterior():
    # prior N(m, s^2), Psi = |x - y|^2 / 2 tilts it to N((m + s^2 y) / (1 + s^2), s^2 / (1 + s^2))
    m, y, s = torch.tensor([2.0, -1.0]), torch.tensor([-1.0, 1.0]), 0.5
    prior = _gaussian_prior(m, s)
    potential = lambda x: 0.5 * ((x - y) ** 2).sum(-1)
    unguided = _sample(prior)
    guided = _sample(_guided(prior, potential))
    post_mean = (m + s**2 * y) / (1 + s**2)
    assert potential(guided).mean() < potential(unguided).mean()
    torch.testing.assert_close(unguided.mean(0), m, atol=0.05, rtol=0)
    torch.testing.assert_close(guided.mean(0), post_mean, atol=0.1, rtol=0)


def test_raises_under_inference_mode():
    guided = _guided(_Field(lambda t, x: -x), lambda x: x.sum(-1))
    with torch.inference_mode(), pytest.raises(RuntimeError, match="autograd"):
        guided.get_vf_fn()(torch.tensor(0.5), torch.randn(2, 3))
