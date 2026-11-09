from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import torch

from sckitflow.core._types import VfFn
from sckitflow.core._utils import broadcast_to_target_shape
from sckitflow.core.nn.fields import NeuralVelocity
from sckitflow.core.probability_paths import BasePath, OneSidedPath

__all__ = [
    "FlowMap",
    "LossGuidedVelocity",
    "MapFn",
    "PathFlowMap",
    "endpoint_coefficients",
    "guidance_strength",
    "posterior_coefficients",
]

MapFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
"""``phi(t, s, x_t) -> x_s``: where a sample at ``x_t`` at time ``t`` is at time ``s``."""

Coefficients = tuple[torch.Tensor, torch.Tensor]


class FlowMap(Protocol):
    """Anything that builds a :data:`MapFn` for given conditions, like ``get_vf_fn`` builds a velocity."""

    def get_map_fn(
        self, condition_dict: dict[str, torch.Tensor] | None = None, source: torch.Tensor | None = None
    ) -> MapFn: ...


def endpoint_coefficients(path: BasePath, t: torch.Tensor) -> tuple[Coefficients, Coefficients]:
    r"""``((a_t, da_t), (beta_t, dbeta_t))`` with $x_t = a_t x_0 + \beta_t x_1$; $a_t = \sigma_t$ on one-sided paths."""
    a_fn = (
        path.compute_sigma_t_and_dot_sigma_t if isinstance(path, OneSidedPath) else path.compute_alpha_t_and_dot_alpha_t
    )
    return a_fn(t), path.compute_beta_t_and_dot_beta_t(t)


def _determinant(path: BasePath, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    r"""$D_t = \dot a_t \beta_t - a_t \dot\beta_t$, and where solving with it gives finite coefficients."""
    (a, da), (beta, dbeta) = endpoint_coefficients(path, t)
    d = da * beta - a * dbeta
    solvable = d.abs() > torch.finfo(d.dtype).eps
    for c in (a, da, beta, dbeta):
        solvable &= torch.isfinite(c / d)
    return d, solvable


def posterior_coefficients(path: BasePath, t: torch.Tensor) -> tuple[Coefficients, Coefficients]:
    r"""``((c0_x, c0_v), (c1_x, c1_v))`` with $\mathbb E[x_i \mid x_t] = c_{i,x} x_t + c_{i,v} v_t$.

    Solving $x_t = a_t x_0 + \beta_t x_1$ and $v_t = \dot a_t x_0 + \dot\beta_t x_1$ gives
    $\hat x_1 = (\dot a_t x_t - a_t v_t) / D_t$ and $\hat x_0 = (\beta_t v_t - \dot\beta_t x_t) / D_t$, with
    $a_t = \sigma_t$ on one-sided paths and $\alpha_t$ otherwise. Where the system is not solvable, e.g. at an
    endpoint where a coefficient is infinite, it returns $\hat x_1 = x_t$ and $\hat x_0 = 0$.
    """
    (a, da), (beta, dbeta) = endpoint_coefficients(path, t)
    d, solvable = _determinant(path, t)
    one, zero = torch.ones_like(d), torch.zeros_like(d)

    def solved(c: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
        return torch.where(solvable, c / d, fallback)

    return (solved(-dbeta, zero), solved(beta, zero)), (solved(da, one), solved(-a, zero))


def guidance_strength(path: BasePath, t: torch.Tensor) -> torch.Tensor:
    r"""$\lambda_t = -a_t D_t / \beta_t$, the factor by which tilting $p_t$ with $e^{-\Psi_t}$ shifts the velocity.

    It is $(1 - t)/t$ on the linear paths and infinite where $\beta_t = 0$; callers clamp it. It assumes $x_0$
    is Gaussian noise, and two-sided paths ignore their $\sigma_t$, so there it is an approximation. It is
    ``0`` where :func:`posterior_coefficients` falls back, which also drops VP's limit of ``1`` at $t = 1$.
    """
    (a, _), (beta, _) = endpoint_coefficients(path, t)
    d, solvable = _determinant(path, t)
    return torch.where(solvable, (-a * d / beta).nan_to_num(nan=0.0, posinf=torch.inf), torch.zeros_like(d))


def _like(c: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Broadcasts a coefficient of ``t``'s shape, a scalar or one per sample, against samples ``x``."""
    return c if c.ndim == 0 else broadcast_to_target_shape(c, x.shape)


class PathFlowMap(torch.nn.Module):
    r"""The flow map of ``velocity``, approximated with ``n_steps`` steps along ``path`` from ``t`` to ``s``.

    Each step from $u$ to $u'$ estimates $\hat x_0, \hat x_1$ from $x_u$ and $v_u$ (see
    :func:`posterior_coefficients`) and re-blends them as $a_{u'} \hat x_0 + \beta_{u'} \hat x_1$. On linear
    paths this is an Euler step, so one step to $s = 1$ gives $x + (1 - t) v_t(x)$, the linear one-step
    estimator. On curved schedules it follows the curve, where an Euler step would not. More steps approach
    the endpoint of the ODE, which a distilled flow map approximates in one network call.

    :param velocity: The velocity module, implementing ``get_vf_fn``.
    :param path: The probability path ``velocity`` was trained on. It must determine $x_0, x_1$ from
        $x_t, v_t$ inside $(0, 1)$, which rules out :class:`EncDecTwoSidedPath`.
    :param n_steps: The number of steps.
    """

    velocity: NeuralVelocity

    def __init__(self, velocity: NeuralVelocity, path: BasePath, n_steps: int = 1) -> None:
        if not _determinant(path, torch.linspace(0.01, 0.99, 99, dtype=torch.float64))[1].all():
            raise ValueError(f"{type(path).__name__} does not determine x_0 and x_1 from x_t and v_t.")
        super().__init__()
        self.velocity, self.path, self.n_steps = velocity, path, n_steps

    def get_map_fn(
        self, condition_dict: dict[str, torch.Tensor] | None = None, source: torch.Tensor | None = None
    ) -> MapFn:
        vf = self.velocity.get_vf_fn(condition_dict=condition_dict, source=source)

        def phi(t: torch.Tensor, s: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            du = (s - t) / self.n_steps
            for k in range(self.n_steps):
                u = t + k * du
                v = vf(u, x)
                (c0_x, c0_v), (c1_x, c1_v) = posterior_coefficients(self.path, u)
                (a_next, _), (beta_next, _) = endpoint_coefficients(self.path, u + du)
                x0 = _like(c0_x, x) * x + _like(c0_v, x) * v
                x1 = _like(c1_x, x) * x + _like(c1_v, x) * v
                x = _like(a_next, x) * x0 + _like(beta_next, x) * x1
            return x

        return phi


class LossGuidedVelocity(torch.nn.Module):
    r"""A prior velocity steered toward low ``potential`` (LabCompass loss guidance), usable as ODE dynamics.

    Its ``get_vf_fn`` returns $v_t(x) - b_t \nabla_x \Psi(\hat x_1(x))$, where $\hat x_1 = \phi_{1 \mid t}(x)$ is
    the ``flow_map``'s estimate of the clean sample and $b_t = $ ``scale`` $\cdot \lambda_t$ clamped to
    ``max_strength`` (see :func:`guidance_strength`). Autograd differentiates through $\hat x_1$, so the
    gradient is $J_{\hat x_1}^\top \nabla \Psi$. Where $b_t = 0$ for every sample, the potential is not
    evaluated. Pass it to :class:`~sckitflow.core.solvers.ODESolver` like any velocity.

    Keep the Jacobian: dropping it and using $\nabla \Psi(\hat x_1)$ alone is cheaper, but with
    ``PathFlowMap(velocity, path)`` $J \to 0$ as $t \to 0$, exactly where $\lambda_t$ is largest, so it is what
    stops the guidance over-steering while $\hat x_1$ is still a guess. Benchmarked against exactly-tilted
    Gaussian mixtures it reached a 2.5-4x lower energy distance to the true posterior, and stayed useful across
    a ~10x range of ``scale`` against a narrow band outside which the Jacobian-free form is worse than not
    guiding at all.

    Needs autograd, so it cannot run under ``torch.inference_mode``. The velocity it returns carries no graph.

    :param velocity: The prior's velocity module, implementing ``get_vf_fn``. $\lambda_t$ assumes the prior
        generates from Gaussian noise, so ``get_vf_fn`` rejects a ``source``.
    :param path: The probability path the prior was trained on.
    :param potential: Maps samples of shape ``(N, D)`` to potentials of shape ``(N,)``, row-wise
        independent, e.g. ``lambda x: surrogate({"dose": x})`` for a :class:`SurrogatePotential`.
    :param flow_map: Estimates $\hat x_1$ as $\phi_{1 \mid t}(x)$, built with the same conditions as the velocity.
    :param scale: Multiplies $\lambda_t$; ``0`` samples the prior without evaluating the potential.
    :param max_strength: Upper bound on $b_t$, which diverges as $\beta_t \to 0$. A flow map's Jacobian does
        not vanish there, so with one this bound sets the early guidance.
    """

    velocity: NeuralVelocity
    potential: Callable[[torch.Tensor], torch.Tensor]

    def __init__(
        self,
        velocity: NeuralVelocity,
        path: BasePath,
        potential: Callable[[torch.Tensor], torch.Tensor],
        *,
        flow_map: FlowMap,
        scale: float = 1.0,
        max_strength: float = 10.0,
    ) -> None:
        super().__init__()
        self.velocity, self.path, self.potential, self.flow_map = velocity, path, potential, flow_map
        self.scale, self.max_strength = scale, max_strength

    def get_vf_fn(
        self,
        condition_dict: dict[str, torch.Tensor] | None = None,
        source: torch.Tensor | None = None,
    ) -> VfFn:
        if source is not None:
            raise ValueError("Loss guidance assumes the prior generates from Gaussian noise, not from a source.")
        vf = self.velocity.get_vf_fn(condition_dict=condition_dict)
        if self.scale == 0:
            return vf
        phi = self.flow_map.get_map_fn(condition_dict=condition_dict)

        def guided(t: torch.Tensor, xt: torch.Tensor) -> torch.Tensor:
            if torch.is_inference_mode_enabled():
                raise RuntimeError("Loss guidance needs autograd; call it outside `torch.inference_mode`.")
            with torch.no_grad():
                v = vf(t, xt)
            strength = _like((self.scale * guidance_strength(self.path, t)).clamp(0, self.max_strength), xt)
            if not (strength > 0).any():
                return v
            with torch.enable_grad():
                x = xt.detach().requires_grad_(True)
                (grad,) = torch.autograd.grad(self.potential(phi(t, torch.ones_like(t), x)).sum(), x)
            # `where`, not a product: a non-finite gradient where the strength is 0 must not leak in
            return v - torch.where(strength > 0, strength * grad, torch.zeros_like(grad))

        return guided
