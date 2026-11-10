from __future__ import annotations

import math
from collections.abc import Callable
from typing import NamedTuple, Protocol

import torch

from sckitflow.core._types import VfFn
from sckitflow.core._utils import broadcast_to_target_shape
from sckitflow.core.nn.fields import NeuralVelocity
from sckitflow.core.probability_paths import BasePath

__all__ = ["FlowMap", "LossGuidedVelocity", "MapFn", "PathFlowMap"]

MapFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
"""``phi(t, s, x_t) -> x_s``: where a sample at ``x_t`` at time ``t`` is at time ``s``."""


class FlowMap(Protocol):
    """Anything that builds a :data:`MapFn` for given conditions, like ``get_vf_fn`` builds a velocity."""

    def get_map_fn(
        self, condition_dict: dict[str, torch.Tensor] | None = None, source: torch.Tensor | None = None
    ) -> MapFn: ...


class _Coefficients(NamedTuple):
    r"""The coefficients of $x_t = \alpha_t x_0 + \beta_t x_1$ and their time derivatives.

    $D_t = \dot\alpha_t \beta_t - \alpha_t \dot\beta_t$ is minus the determinant of the system for $x_0, x_1$.
    """

    alpha: torch.Tensor
    dalpha: torch.Tensor
    beta: torch.Tensor
    dbeta: torch.Tensor
    D: torch.Tensor


def _coefficients(path: BasePath, t: torch.Tensor) -> _Coefficients:
    (alpha, dalpha), (beta, dbeta) = path.compute_alpha_t_and_dot_alpha_t(t), path.compute_beta_t_and_dot_beta_t(t)
    return _Coefficients(alpha, dalpha, beta, dbeta, dalpha * beta - alpha * dbeta)


def _posterior_means(c: _Coefficients, x: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    r"""$(\hat x_0, \hat x_1)$, solving $x_t = \alpha_t x_0 + \beta_t x_1$ and $v_t = \dot\alpha_t x_0 + \dot\beta_t x_1$."""

    def solved(numerator: torch.Tensor) -> torch.Tensor:
        return broadcast_to_target_shape(numerator / c.D, x.shape)

    x0 = solved(-c.dbeta) * x + solved(c.beta) * v
    x1 = solved(c.dalpha) * x + solved(-c.alpha) * v
    return x0, x1


def _check_path(path: BasePath) -> None:
    r"""Raises unless ``path`` determines $x_0, x_1$ from $x_t, v_t$ at every $t$: noiseless, with $D_t \neq 0$."""
    if path.sigma > 0:
        raise ValueError("A path with sigma > 0 does not determine x_0 and x_1 from x_t and v_t.")
    c = _coefficients(path, torch.linspace(0, 1, 101, dtype=torch.float64))
    if not (torch.isfinite(torch.stack(c)).all() and (c.D.abs() > torch.finfo(c.D.dtype).eps).all()):
        raise ValueError(f"{type(path).__name__} does not determine x_0 and x_1 from x_t and v_t.")


def _guidance_strength(path: BasePath, t: torch.Tensor) -> torch.Tensor:
    r"""$\lambda_t = -\alpha_t D_t / \beta_t$, the factor by which tilting $p_t$ with $e^{-\Psi_t}$ shifts the velocity.

    It is $(1 - t)/t$ on the linear path and infinite where $\beta_t = 0$; callers clamp it. It assumes $x_0$
    is Gaussian noise. It is ``0`` where $\alpha_t$ is zero up to rounding, the data end, where e.g.
    $\cos(\pi/2)$ would otherwise leave a positive $10^{-16}$ in float64.
    """
    c = _coefficients(path, t)
    active = c.alpha.abs() > torch.finfo(c.alpha.dtype).eps
    return torch.where(active, -c.alpha * c.D / c.beta, torch.zeros_like(c.D))


class PathFlowMap(torch.nn.Module):
    r"""The flow map of ``velocity``, approximated with ``n_steps`` steps along ``path`` from ``t`` to ``s``.

    Each step from $u$ to $u'$ estimates the posterior means $\hat x_0, \hat x_1$ from $x_u$ and $v_u$ and
    re-blends them as $\alpha_{u'} \hat x_0 + \beta_{u'} \hat x_1$. On linear paths this is an Euler step, so one step
    to $s = 1$ gives $x + (1 - t) v_t(x)$, the linear one-step estimator. On curved schedules it follows the
    curve, where an Euler step would not. More steps approach the endpoint of the ODE, which a distilled flow
    map approximates in one network call.

    :param velocity: The velocity module, implementing ``get_vf_fn``.
    :param path: The probability path ``velocity`` was trained on. It must determine $x_0, x_1$ from
        $x_t, v_t$ inside $(0, 1)$, which rules out :class:`EncDecPath` and noisy paths.
    :param n_steps: The number of steps, at least one.
    """

    velocity: NeuralVelocity

    def __init__(self, velocity: NeuralVelocity, path: BasePath, n_steps: int = 1) -> None:
        if n_steps < 1:
            raise ValueError(f"n_steps must be at least 1, got {n_steps}.")
        _check_path(path)
        super().__init__()
        self.velocity, self.path, self.n_steps = velocity, path, n_steps

    def get_map_fn(
        self, condition_dict: dict[str, torch.Tensor] | None = None, source: torch.Tensor | None = None
    ) -> MapFn:
        vf = self.velocity.get_vf_fn(condition_dict=condition_dict, source=source)

        def phi(t: torch.Tensor, s: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            t, s = t.to(x.dtype), s.to(x.dtype)
            for k in range(self.n_steps):
                # lerp is exact at weight 1, so the last step ends on s even in float32
                u = torch.lerp(t, s, k / self.n_steps)
                x0, x1 = _posterior_means(_coefficients(self.path, u), x, vf(u, x))
                after = _coefficients(self.path, torch.lerp(t, s, (k + 1) / self.n_steps))
                x = (
                    broadcast_to_target_shape(after.alpha, x.shape) * x0
                    + broadcast_to_target_shape(after.beta, x.shape) * x1
                )
            return x

        return phi


class LossGuidedVelocity(torch.nn.Module):
    r"""A prior velocity steered toward low ``potential`` (LabCompass loss guidance), usable as ODE dynamics.

    Its ``get_vf_fn`` returns $v_t(x) - b_t \nabla_x \Psi(\hat x_1(x))$, where $\hat x_1 = \phi_{1 \mid t}(x)$ is
    the ``flow_map``'s estimate of the clean sample and $b_t = $ ``scale`` $\cdot \lambda_t$ clamped to
    ``max_strength``, with $\lambda_t = -\alpha_t D_t / \beta_t$ from the path, $(1 - t)/t$ on the linear path.
    Autograd differentiates through $\hat x_1$, so the gradient is $J_{\hat x_1}^\top \nabla \Psi$. Where
    $b_t = 0$ for every sample, the potential is not evaluated. Pass it to
    :class:`~sckitflow.core.solvers.ODESolver` like any velocity.

    Keep the Jacobian: dropping it and using $\nabla \Psi(\hat x_1)$ alone is cheaper, but with
    ``PathFlowMap(velocity, path)`` and its default single step, $\hat x_1$ is the posterior mean, whose
    Jacobian $J \to 0$ as $t \to 0$, exactly where $\lambda_t$ is largest, so it is what stops the guidance
    over-steering while $\hat x_1$ is still a guess. More steps, or a distilled flow map, approach the ODE's own
    endpoint, whose Jacobian does not vanish; then ``max_strength`` sets the early guidance.

    Each step evaluates the prior twice, once for $v_t$ and once inside the flow map: the flow map is opaque, so
    its own evaluation cannot be reused.

    Needs autograd, so it cannot run under ``torch.inference_mode``, which rules out ``predict_adata``; solve it
    with :class:`~sckitflow.core.solvers.ODESolver` directly. The velocity it returns carries no graph.

    :param velocity: The prior's velocity module, implementing ``get_vf_fn``. $\lambda_t$ assumes the prior
        generates from Gaussian noise, so ``get_vf_fn`` rejects a ``source``.
    :param path: The probability path the prior was trained on. Paths :class:`PathFlowMap` rejects are rejected
        here too.
    :param potential: Maps samples of shape ``(N, D)`` to potentials of shape ``(N,)``, row-wise
        independent, e.g. ``lambda x: surrogate({"dose": x})`` for a :class:`SurrogatePotential`.
    :param flow_map: A flow map of the same prior, estimating $\hat x_1 = \phi_{1 \mid t}(x)$.
    :param scale: Multiplies $\lambda_t$, finite and non-negative; ``0`` returns the prior's velocity unchanged.
    :param max_strength: Finite non-negative upper bound on $b_t$, which diverges as $\beta_t \to 0$.
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
        if not (math.isfinite(scale) and math.isfinite(max_strength) and scale >= 0 and max_strength >= 0):
            raise ValueError(f"scale and max_strength must be finite and non-negative, got {scale} and {max_strength}.")
        _check_path(path)
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
            t = t.to(xt.dtype)
            with torch.no_grad():
                v = vf(t, xt)
            strength = (self.scale * _guidance_strength(self.path, t)).clamp(0, self.max_strength)
            strength = broadcast_to_target_shape(strength, xt.shape)
            if not (strength > 0).any():
                return v
            with torch.enable_grad():
                x = xt.detach().requires_grad_(True)
                (grad,) = torch.autograd.grad(self.potential(phi(t, torch.ones_like(t), x)).sum(), x)
            # `where`, not a product: a non-finite gradient where the strength is 0 must not leak in
            return v - torch.where(strength > 0, strength * grad, torch.zeros_like(grad))

        return guided
