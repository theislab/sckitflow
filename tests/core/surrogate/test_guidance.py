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
from sckitflow.core.surrogate import LossGuidedVelocity, PathFlowMap
from sckitflow.core.surrogate._guidance import _coefficients, _guidance_strength, _posterior_means

PATHS = [LinearOneSidedPath, TrigOneSidedPath, VPOneSidedPath, LinearTwoSidedPath, TrigTwoSidedPath, EncDecTwoSidedPath]
SOLVABLE = [p for p in PATHS if p is not EncDecTwoSidedPath]


class _Field(torch.nn.Module):
    """A velocity module wrapping a plain ``v(t, x)``."""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def get_vf_fn(self, condition_dict=None, source=None):
        return self.fn


def _path(path_cls):
    """One-sided paths need noise; two-sided ones are deterministic here."""
    if issubclass(path_cls, OneSidedPath):
        return path_cls(sigma=1.0, sampler_cfg=IsotropicGaussianNoiseSamplerConfig())
    return path_cls(sigma=0.0)


def _guided(field, potential, path=None, **kwargs):
    path = _path(LinearOneSidedPath) if path is None else path
    return LossGuidedVelocity(field, path, potential, flow_map=PathFlowMap(field, path), **kwargs)


@pytest.mark.parametrize("path_cls", SOLVABLE)
@pytest.mark.parametrize("t", [0.1, 0.3, 0.7, 0.9])
def test_posterior_means_recover_the_pair(path_cls, t):
    # draws from the path's own sampler, with noise on the one-sided paths, so the assumed blend is checked
    one_sided, path = issubclass(path_cls, OneSidedPath), _path(path_cls)
    t, generator = torch.tensor(t, dtype=torch.float64), torch.Generator().manual_seed(0)
    x0, x1 = torch.randn(2, 8, 3, dtype=torch.float64)
    xt = path.compute_xt(t, x0, x1, generator=generator)
    x0_hat, x1_hat = _posterior_means(_coefficients(path, t), xt, path.compute_ut(t, xt, x0, x1))
    torch.testing.assert_close(x1_hat, x1)
    if not one_sided:  # one-sided paths draw their own x0
        torch.testing.assert_close(x0_hat, x0)
    assert _guidance_strength(path, t) >= 0


def test_closed_forms():
    t = torch.linspace(0.05, 0.95, 7, dtype=torch.float64)
    x, v = torch.ones_like(t), torch.full_like(t, 2.0)
    path = _path(LinearOneSidedPath)
    torch.testing.assert_close(_posterior_means(_coefficients(path, t), x, v)[1], x + (1 - t) * v)
    torch.testing.assert_close(_guidance_strength(path, t), (1 - t) / t)
    torch.testing.assert_close(
        _posterior_means(_coefficients(_path(VPOneSidedPath), t), x, v)[1], t * x + (1 - t**2) * v
    )


@pytest.mark.parametrize(
    "path",
    [_path(EncDecTwoSidedPath), LinearTwoSidedPath(sigma=1.0, sampler_cfg=IsotropicGaussianNoiseSamplerConfig())],
    ids=["encdec", "noisy-two-sided"],
)
def test_path_flow_map_rejects_unsolvable_paths(path):
    with pytest.raises(ValueError, match="does not determine"):
        PathFlowMap(_Field(lambda t, x: -x), path)


def test_path_flow_map_rejects_no_steps():
    with pytest.raises(ValueError, match="at least 1"):
        PathFlowMap(_Field(lambda t, x: -x), _path(LinearOneSidedPath), n_steps=0)


@pytest.mark.parametrize("path_cls", SOLVABLE)
def test_zero_strength_rows_ignore_a_nan_gradient(path_cls):
    # per-sample t: row 0 is guided, row 1 sits at t = 1 where the strength is 0 and its gradient is NaN
    potential = lambda x: (x**2).sum(-1) + torch.sqrt(50 - x[:, 0])  # NaN, with a NaN gradient, where x_0 > 50
    t, x = torch.tensor([0.5, 1.0]), torch.tensor([[1.0, 1.0, 1.0], [100.0, 1.0, 1.0]])
    velocity = _guided(_Field(lambda t, x: -x), potential, _path(path_cls)).get_vf_fn()(t, x)
    assert torch.isfinite(velocity).all()
    torch.testing.assert_close(velocity[1], -x[1])
    assert not torch.allclose(velocity[0], -x[0])


@pytest.mark.parametrize(
    "bad", [{"scale": float("inf")}, {"max_strength": float("inf")}, {"scale": -1.0}, {"max_strength": -1.0}]
)
def test_rejects_non_finite_or_negative_strength(bad):
    with pytest.raises(ValueError, match="finite and non-negative"):
        _guided(_Field(lambda t, x: -x), lambda x: x.sum(-1), **bad)


def test_rejects_paths_without_guidance():
    field = _Field(lambda t, x: -x)
    with pytest.raises(ValueError, match="does not determine"):
        LossGuidedVelocity(
            field,
            _path(EncDecTwoSidedPath),
            lambda x: x.sum(-1),
            flow_map=PathFlowMap(field, _path(LinearOneSidedPath)),
        )


def test_output_dtype_follows_samples():
    v = _guided(_Field(lambda t, x: -x), lambda x: (x**2).sum(-1)).get_vf_fn()(
        torch.tensor(0.5, dtype=torch.float64), torch.ones(2, 3)
    )
    assert v.dtype == torch.float32


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
    path, t = _path(path_cls), torch.tensor(0.3, dtype=torch.float64)
    x0, x1 = torch.randn(2, 8, 3, dtype=torch.float64)

    def v(u, x):
        c = _coefficients(path, u)
        return c.da * x0 + c.dbeta * x1

    c = _coefficients(path, t)
    a, b = c.a, c.beta
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
