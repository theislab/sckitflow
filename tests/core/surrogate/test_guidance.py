import pytest
import torch
from tests.core.surrogate.shared import Field

from sckitflow import IsotropicGaussianNoiseSamplerConfig
from sckitflow.core.probability_paths import EncDecPath, LinearPath, TrigPath
from sckitflow.core.solvers import ODESolver
from sckitflow.core.surrogate import LossGuidedVelocity, PathFlowMap
from sckitflow.core.surrogate._guidance import _compute_guidance_strength, _compute_posterior_means

SOLVABLE = [LinearPath, TrigPath]  # noiseless; EncDecPath needs sigma > 0, which guidance rejects
_NOISE = IsotropicGaussianNoiseSamplerConfig()


def _path(path_cls):
    return path_cls(sigma=0.0)


def _guided(field, potential, path=None, **kwargs):
    path = _path(LinearPath) if path is None else path
    return LossGuidedVelocity(field, path, potential, flow_map=PathFlowMap(field, path), **kwargs)


@pytest.mark.parametrize("path_cls", SOLVABLE)
@pytest.mark.parametrize("t", [0.1, 0.3, 0.7, 0.9])
def test_posterior_means_recover_the_pair(path_cls, t):
    # draws from the path's own sampler, so the assumed blend is checked against the path itself
    path = _path(path_cls)
    t, generator = torch.tensor(t, dtype=torch.float64), torch.Generator().manual_seed(0)
    x0, x1 = torch.randn(2, 8, 3, dtype=torch.float64)
    xt = path.compute_xt(t, x0, x1, generator=generator)
    x0_hat, x1_hat = _compute_posterior_means(path, t, xt, path.compute_ut(t, xt, x0, x1))
    torch.testing.assert_close(x1_hat, x1)
    torch.testing.assert_close(x0_hat, x0)


def test_closed_forms():
    t = torch.linspace(0.05, 0.95, 7, dtype=torch.float64)
    x, v = torch.ones_like(t), torch.full_like(t, 2.0)
    path = _path(LinearPath)
    torch.testing.assert_close(_compute_posterior_means(path, t, x, v)[1], x + (1 - t) * v)
    torch.testing.assert_close(_compute_guidance_strength(path, t), (1 - t) / t)
    # trig: alpha = cos, beta = sin, so D = -pi/2 and lambda = (pi/2) cot(pi t / 2)
    torch.testing.assert_close(
        _compute_guidance_strength(_path(TrigPath), t), (torch.pi / 2) / torch.tan(torch.pi * t / 2)
    )


@pytest.mark.parametrize(
    "path",
    [EncDecPath(sigma=1.0, sampler_cfg=_NOISE), LinearPath(sigma=1.0, sampler_cfg=_NOISE)],
    ids=["encdec", "noisy-linear"],
)
def test_path_flow_map_rejects_unsolvable_paths(path):
    with pytest.raises(ValueError, match="does not determine"):
        PathFlowMap(Field(lambda t, x: -x), path)


def test_path_flow_map_rejects_no_steps():
    with pytest.raises(ValueError, match="at least 1"):
        PathFlowMap(Field(lambda t, x: -x), _path(LinearPath), n_steps=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("path_cls", SOLVABLE)
def test_zero_strength_rows_ignore_a_nan_gradient(path_cls, dtype):
    # per-sample t: row 0 is guided, row 1 sits at t = 1 where the strength is 0 and its gradient is NaN
    potential = lambda x: (x**2).sum(-1) + torch.sqrt(50 - x[:, 0])  # NaN, with a NaN gradient, where x_0 > 50
    t = torch.tensor([0.5, 1.0], dtype=dtype)
    x = torch.tensor([[1.0, 1.0, 1.0], [100.0, 1.0, 1.0]], dtype=dtype)
    velocity = _guided(Field(lambda t, x: -x), potential, _path(path_cls)).get_vf_fn()(t, x)
    assert torch.isfinite(velocity).all()
    torch.testing.assert_close(velocity[1], -x[1])
    assert not torch.allclose(velocity[0], -x[0])


@pytest.mark.parametrize(
    "bad", [{"scale": float("inf")}, {"max_strength": float("inf")}, {"scale": -1.0}, {"max_strength": -1.0}]
)
def test_rejects_non_finite_or_negative_strength(bad):
    with pytest.raises(ValueError, match="finite and non-negative"):
        _guided(Field(lambda t, x: -x), lambda x: x.sum(-1), **bad)


def test_rejects_paths_without_guidance():
    field = Field(lambda t, x: -x)
    with pytest.raises(ValueError, match="does not determine"):
        LossGuidedVelocity(
            field,
            EncDecPath(sigma=1.0, sampler_cfg=_NOISE),
            lambda x: x.sum(-1),
            flow_map=PathFlowMap(field, _path(LinearPath)),
        )


def test_output_dtype_follows_samples():
    v = _guided(Field(lambda t, x: -x), lambda x: (x**2).sum(-1)).get_vf_fn()(
        torch.tensor(0.5, dtype=torch.float64), torch.ones(2, 3)
    )
    assert v.dtype == torch.float32


def test_zero_strength_skips_the_potential():
    guided = _guided(Field(lambda t, x: -x), lambda x: pytest.fail("evaluated"))
    torch.testing.assert_close(guided.get_vf_fn()(torch.tensor(1.0), torch.ones(2, 3)), -torch.ones(2, 3))


def test_per_sample_time():
    field, potential = Field(lambda t, x: -x), lambda x: (x**2).sum(-1)
    t, x = torch.tensor([0.2, 0.5, 0.8, 1.0]), torch.randn(4, 3)
    batched = _guided(field, potential).get_vf_fn()(t, x)
    for i in range(4):
        torch.testing.assert_close(batched[i], _guided(field, potential).get_vf_fn()(t[i], x[i : i + 1])[0])


def test_rejects_a_source():
    with pytest.raises(ValueError, match="noise"):
        _guided(Field(lambda t, x: -x), lambda x: x.sum(-1)).get_vf_fn(source=torch.zeros(2, 3))


@pytest.mark.parametrize(
    ("t", "scale", "max_strength", "strength"),
    [(0.4, 0.5, 10.0, 0.75), (0.05, 1.0, 2.0, 2.0)],  # (1 - t) / t times scale, then capped
    ids=["scaled", "capped"],
)
def test_gradient_matches_analytic(t, scale, max_strength, strength):
    # v = A x + c is linear, so x1_hat = M x + (1 - t) c with M = I + (1 - t) A and the guidance is
    # -b_t M^T grad Psi(x1_hat) for Psi(x) = |x - y|^2 / 2.
    torch.manual_seed(0)
    A, c, y = (
        torch.randn(3, 3, dtype=torch.float64),
        torch.randn(3, dtype=torch.float64),
        torch.randn(3, dtype=torch.float64),
    )
    t, x = torch.tensor(t, dtype=torch.float64), torch.randn(5, 3, dtype=torch.float64)
    field = Field(lambda t, x: x @ A.T + c)
    guided = _guided(field, lambda x: 0.5 * ((x - y) ** 2).sum(-1), scale=scale, max_strength=max_strength)
    M = torch.eye(3, dtype=torch.float64) + (1 - t) * A
    x1_hat = x @ M.T + (1 - t) * c
    expected = x @ A.T + c - strength * (x1_hat - y) @ M
    torch.testing.assert_close(guided.get_vf_fn()(t, x), expected)


@pytest.mark.parametrize(("dtype", "t"), [(torch.float64, 0.3), (torch.float32, 0.21)])
@pytest.mark.parametrize("path_cls", SOLVABLE)
def test_path_flow_map_follows_curved_paths(path_cls, dtype, t):
    # the path's own conditional velocity for one pair: several steps land on x1 however curved the schedule,
    # where plain Euler steps would not on the trig path, in float32 as in float64
    path, t = _path(path_cls), torch.tensor(t, dtype=dtype)
    x0, x1 = torch.randn(2, 8, 3, dtype=dtype)
    xt = path.compute_xt(t, x0, x1, generator=torch.Generator().manual_seed(0))
    phi = PathFlowMap(Field(lambda u, x: path.compute_ut(u, x, x0, x1)), path, n_steps=3).get_map_fn()
    torch.testing.assert_close(phi(t, torch.ones_like(t), xt), x1)


def _gaussian_prior(m, s):
    """Exact marginal velocity of the noiseless linear path from Gaussian noise to data ``N(m, s^2 I)``."""

    def v(t, x):
        x1_hat = m + t * s**2 / (t**2 * s**2 + (1 - t) ** 2) * (x - t * m)
        return (x1_hat - x) / (1 - t)

    return Field(v)


def _sample(field, n=4000, steps=100):
    x0 = torch.randn(n, 2, generator=torch.Generator().manual_seed(0))
    return ODESolver(field).solve(x0, torch.linspace(0, 1 - 1e-3, steps))


def test_scale_zero_is_the_prior():
    prior = Field(lambda t, x: -x)
    assert _guided(prior, lambda x: x.sum(-1), scale=0.0).get_vf_fn() is prior.fn


def test_guidance_descends_toward_posterior():
    # prior N(m, s^2), Psi = |x - y|^2 / 2 tilts it to N((m + s^2 y) / (1 + s^2), s^2 / (1 + s^2)). Guidance only
    # approximates the tilt: its mean is off by a fixed ~0.08 here, so the tolerance bounds that bias, and the
    # analytic-gradient test pins the strength exactly.
    m, y, s = torch.tensor([2.0, -1.0]), torch.tensor([-1.0, 1.0]), 0.5
    prior = _gaussian_prior(m, s)
    potential = lambda x: 0.5 * ((x - y) ** 2).sum(-1)
    unguided = _sample(prior)
    guided = _sample(_guided(prior, potential))
    post_mean = (m + s**2 * y) / (1 + s**2)
    assert potential(guided).mean() < potential(unguided).mean()
    torch.testing.assert_close(unguided.mean(0), m, atol=0.05, rtol=0)
    torch.testing.assert_close(guided.mean(0), post_mean, atol=0.1, rtol=0)
    torch.testing.assert_close(guided.var(0), torch.full((2,), s**2 / (1 + s**2)), atol=0.03, rtol=0)


def test_raises_under_inference_mode():
    guided = _guided(Field(lambda t, x: -x), lambda x: x.sum(-1))
    with torch.inference_mode(), pytest.raises(RuntimeError, match="autograd"):
        guided.get_vf_fn()(torch.tensor(0.5), torch.randn(2, 3))
