from __future__ import annotations

import math
from collections.abc import Callable
from typing import Protocol

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


def _compute_coefficients(path: BasePath, t: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """``(alpha, dalpha, beta, dbeta)`` at ``t``."""
    return (*path.compute_alpha_t_and_dot_alpha_t(t), *path.compute_beta_t_and_dot_beta_t(t))


def _compute_posterior_means(
    path: BasePath, t: torch.Tensor, x: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Posterior means ``(x0, x1)``, solving ``x = alpha x0 + beta x1`` and ``v = dalpha x0 + dbeta x1``."""
    a, da, b, db = (broadcast_to_target_shape(c, x.shape) for c in _compute_coefficients(path, t))
    det = a * db - b * da
    return (db * x - b * v) / det, (a * v - da * x) / det


def _compute_guidance_strength(path: BasePath, t: torch.Tensor) -> torch.Tensor:
    """The guidance strength ``alpha * det / beta``; 0 where ``alpha`` rounds to 0, at the data end."""
    a, da, b, db = _compute_coefficients(path, t)
    strength = a * (a * db - b * da) / b
    return torch.where(a.abs() > torch.finfo(a.dtype).eps, strength, torch.zeros_like(strength))


def _check_path(path: BasePath) -> None:
    """Raises unless ``path`` is noiseless and its ``x_t, v_t`` determine ``x0, x1`` at every ``t``."""
    if path.sigma > 0:
        raise ValueError("A path with sigma > 0 does not determine x_0 and x_1 from x_t and v_t.")
    a, da, b, db = _compute_coefficients(path, torch.linspace(0, 1, 101, dtype=torch.float64))
    det = a * db - b * da
    if not (torch.isfinite(det).all() and (det.abs() > torch.finfo(det.dtype).eps).all()):
        raise ValueError(f"{type(path).__name__} does not determine x_0 and x_1 from x_t and v_t.")


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
                x0, x1 = _compute_posterior_means(self.path, u, x, vf(u, x))
                x = self.path.compute_xt(torch.lerp(t, s, (k + 1) / self.n_steps), x0, x1, generator=None)
            return x

        return phi


class LossGuidedVelocity(torch.nn.Module):
    r"""A prior velocity steered toward low ``potential`` (LabCompass loss guidance), usable as ODE dynamics.

    Its ``get_vf_fn`` returns $v_t(x) - b_t \nabla_x \Psi(\hat x_1(x))$, where $\hat x_1 = \phi_{1 \mid t}(x)$ is
    the ``flow_map``'s estimate of the clean sample and $b_t = $ ``scale`` $\cdot \lambda_t$ clamped to
    ``max_strength``, with $\lambda_t = \alpha_t (\alpha_t \dot\beta_t - \beta_t \dot\alpha_t) / \beta_t$ from the path,
    $(1 - t)/t$ on the linear path.
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
            strength = (self.scale * _compute_guidance_strength(self.path, t)).clamp(0, self.max_strength)
            strength = broadcast_to_target_shape(strength, xt.shape)
            if not (strength > 0).any():
                return v
            with torch.enable_grad():
                x = xt.detach().requires_grad_(True)
                (grad,) = torch.autograd.grad(self.potential(phi(t, torch.ones_like(t), x)).sum(), x)
            # `where`, not a product: a non-finite gradient where the strength is 0 must not leak in
            return v - torch.where(strength > 0, strength * grad, torch.zeros_like(grad))

        return guided
