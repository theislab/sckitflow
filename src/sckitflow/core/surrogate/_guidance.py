from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import torch

from sckitflow.core._types import VfFn
from sckitflow.core.nn.fields import NeuralVelocity
from sckitflow.core.probability_paths import BasePath, OneSidedPath

__all__ = ["FlowMap", "LossGuidedVelocity", "MapFn", "PathFlowMap", "guidance_coefficients"]

MapFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
"""``phi(t, s, x_t) -> x_s``: where a sample at ``x_t`` at time ``t`` is at time ``s``."""


class FlowMap(Protocol):
    """Anything that builds a :data:`MapFn` for given conditions, like ``get_vf_fn`` builds a velocity."""

    def get_map_fn(
        self, condition_dict: dict[str, torch.Tensor] | None = None, source: torch.Tensor | None = None
    ) -> MapFn: ...


class PathFlowMap(torch.nn.Module):
    r"""The flow map of ``velocity``, approximated with ``n_steps`` steps along ``path`` from ``t`` to ``s``.

    Each step from $u$ to $u'$ solves $x_u = a_u x_0 + \beta_u x_1$ and $v_u = \dot a_u x_0 + \dot\beta_u x_1$ for the
    posterior means $\hat x_0, \hat x_1$ (see :func:`guidance_coefficients`) and re-blends them as
    $a_{u'} \hat x_0 + \beta_{u'} \hat x_1$. On linear paths this is an Euler step, so one step to $s = 1$ gives
    $x + (1 - t) v_t(x)$, the linear one-step estimator. On curved schedules it follows the curve, where an
    Euler step would not. More steps approach the endpoint of the ODE, which a distilled flow map approximates
    in one network call.

    :param velocity: The velocity module, implementing ``get_vf_fn``.
    :param path: The probability path ``velocity`` was trained on.
    :param n_steps: The number of steps.
    """

    velocity: NeuralVelocity

    def __init__(self, velocity: NeuralVelocity, path: BasePath, n_steps: int = 1) -> None:
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
                c_x, c_v, _ = guidance_coefficients(self.path, u)
                x1 = c_x * x + c_v * vf(u, x)
                (a, _), (b, _) = _blend(self.path, u)
                (a_next, _), (b_next, _) = _blend(self.path, u + du)
                # a = 0 only at the data end, where x0 drops out (a_next = 0 too)
                x0 = (x - b * x1) / torch.where(a == 0, torch.ones_like(a), a)
                x = a_next * x0 + b_next * x1
            return x

        return phi


def _blend(
    path: BasePath, t: torch.Tensor
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    r"""``((a_t, da_t), (beta_t, dbeta_t))`` with $x_t = a_t x_0 + \beta_t x_1$; $a_t = \sigma_t$ on one-sided paths."""
    a_fn = (
        path.compute_sigma_t_and_dot_sigma_t if isinstance(path, OneSidedPath) else path.compute_alpha_t_and_dot_alpha_t
    )
    return a_fn(t), path.compute_beta_t_and_dot_beta_t(t)


def guidance_coefficients(path: BasePath, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Returns ``(c_x, c_v, lam)`` so that $\hat x_1 = c_x x_t + c_v v_t$ and the guidance strength is ``lam``.

    With $x_t = a_t x_0 + \beta_t x_1$, where $a_t = \sigma_t$ on one-sided paths and $\alpha_t$ otherwise,
    solving $x_t$ and $v_t = \dot a_t x_0 + \dot\beta_t x_1$ for $x_1$ gives $c_x = \dot a_t / D_t$ and
    $c_v = -a_t / D_t$ with $D_t = \dot a_t \beta_t - a_t \dot\beta_t$. ``lam`` $= -a_t D_t / \beta_t$ is the
    factor by which tilting $p_t$ with $e^{-\Psi_t}$ shifts a Gaussian path's velocity, $(1 - t)/t$ on the
    linear one-sided path. Two-sided paths ignore their $\sigma_t$, so there it is an approximation.

    Where $x_1$ is not identifiable ($D_t = 0$) or a coefficient blows up at an endpoint, this returns
    $\hat x_1 = x_t$ with ``lam`` $= 0$, which switches guidance off on :class:`EncDecTwoSidedPath`.
    ``lam`` is infinite where $\beta_t = 0$; callers clamp it.
    """
    (a, da), (beta, dbeta) = _blend(path, t)
    d = da * beta - a * dbeta
    c_x, c_v = da / d, -a / d
    ok = torch.isfinite(c_x) & torch.isfinite(c_v) & (d.abs() > torch.finfo(d.dtype).eps)
    c_x, c_v = torch.where(ok, c_x, torch.ones_like(c_x)), torch.where(ok, c_v, torch.zeros_like(c_v))
    lam = torch.where(ok, (-a * d / beta).nan_to_num(nan=0.0, posinf=torch.inf), torch.zeros_like(d))
    return c_x, c_v, lam


class LossGuidedVelocity(torch.nn.Module):
    r"""A prior velocity steered toward low ``potential`` (LabCompass loss guidance), usable as ODE dynamics.

    Its ``get_vf_fn`` returns $v_t(x) - b_t \nabla_x \Psi(\hat x_1(x))$, where $\hat x_1 = \phi_{1 \mid t}(x)$ is
    the ``flow_map``'s estimate of the clean sample and $b_t = $ ``scale`` $\cdot \lambda_t$ clamped to
    ``max_strength`` (see :func:`guidance_coefficients`). Autograd differentiates through $\hat x_1$, so the
    gradient is $J_{\hat x_1}^\top \nabla \Psi$. Pass it to :class:`~sckitflow.core.solvers.ODESolver` like any
    velocity.

    Keep the Jacobian: dropping it and using $\nabla \Psi(\hat x_1)$ alone is cheaper, but with
    ``PathFlowMap(velocity, path)`` $J \to 0$ as $t \to 0$, exactly where $\lambda_t$ is largest, so it is what stops the guidance over-steering while
    $\hat x_1$ is still a guess. Benchmarked against exactly-tilted Gaussian mixtures it reached a 2.5-4x
    lower energy distance to the true posterior, and stayed useful across a ~10x range of ``scale`` against
    a narrow band outside which the Jacobian-free form is worse than not guiding at all.

    Needs autograd, so it cannot run under ``torch.inference_mode``.

    :param velocity: The prior's velocity module, implementing ``get_vf_fn``.
    :param path: The probability path the prior was trained on.
    :param potential: Maps samples of shape ``(N, D)`` to potentials of shape ``(N,)``, row-wise
        independent, e.g. ``lambda x: surrogate({"dose": x})`` for a :class:`SurrogatePotential`.
    :param scale: Multiplies $\lambda_t$; ``0`` samples the prior without evaluating the potential.
    :param max_strength: Upper bound on $b_t$, which diverges as $\beta_t \to 0$. A flow map's Jacobian does
        not vanish there, so with one this bound sets the early guidance.
    :param flow_map: Estimates $\hat x_1$ as $\phi_{1 \mid t}(x)$, built with the same conditions as the velocity.
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
        vf = self.velocity.get_vf_fn(condition_dict=condition_dict, source=source)
        if self.scale == 0:
            return vf
        phi = self.flow_map.get_map_fn(condition_dict=condition_dict, source=source)

        def guided(t: torch.Tensor, xt: torch.Tensor) -> torch.Tensor:
            if torch.is_inference_mode_enabled():
                raise RuntimeError("Loss guidance needs autograd; call it outside `torch.inference_mode`.")
            *_, lam = guidance_coefficients(self.path, t)
            with torch.enable_grad():
                x = xt.detach().requires_grad_(True)
                (grad,) = torch.autograd.grad(self.potential(phi(t, torch.ones_like(t), x)).sum(), x)
            return vf(t, xt).detach() - (self.scale * lam).clamp(0, self.max_strength) * grad

        return guided
