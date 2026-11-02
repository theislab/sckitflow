from __future__ import annotations

from collections.abc import Callable

import torch

from sckitflow.core._types import VfFn
from sckitflow.core.nn.fields import NeuralVelocity
from sckitflow.core.probability_paths import BasePath, OneSidedPath

__all__ = ["LossGuidedVelocity", "guidance_coefficients"]


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
    beta, dbeta = path.compute_beta_t_and_dot_beta_t(t)
    a, da = (
        path.compute_sigma_t_and_dot_sigma_t if isinstance(path, OneSidedPath) else path.compute_alpha_t_and_dot_alpha_t
    )(t)
    d = da * beta - a * dbeta
    c_x, c_v = da / d, -a / d
    ok = torch.isfinite(c_x) & torch.isfinite(c_v) & (d.abs() > torch.finfo(d.dtype).eps)
    c_x, c_v = torch.where(ok, c_x, torch.ones_like(c_x)), torch.where(ok, c_v, torch.zeros_like(c_v))
    lam = torch.where(ok, (-a * d / beta).nan_to_num(nan=0.0, posinf=torch.inf), torch.zeros_like(d))
    return c_x, c_v, lam


class LossGuidedVelocity(torch.nn.Module):
    r"""A prior velocity steered toward low ``potential`` (LabCompass loss guidance), usable as ODE dynamics.

    Its ``get_vf_fn`` returns $v_t(x) - b_t \nabla_x \Psi(\hat x_1(x))$, where $\hat x_1$ is the one-step
    estimate of the clean sample from the path's coefficients and $b_t = $ ``scale`` $\cdot \lambda_t$
    clamped to ``max_strength`` (see :func:`guidance_coefficients`). Autograd differentiates through
    $\hat x_1$, so the gradient is $J_{\hat x_1}^\top \nabla \Psi$. Pass it to
    :class:`~sckitflow.core.solvers.ODESolver` like any velocity.

    Keep the Jacobian: dropping it and using $\nabla \Psi(\hat x_1)$ alone is cheaper, but $J \to 0$ as
    $t \to 0$, exactly where $\lambda_t$ is largest, so it is what stops the guidance over-steering while
    $\hat x_1$ is still a guess. Benchmarked against exactly-tilted Gaussian mixtures it reached a 2.5-4x
    lower energy distance to the true posterior, and stayed useful across a ~10x range of ``scale`` against
    a narrow band outside which the Jacobian-free form is worse than not guiding at all.

    Needs autograd, so it cannot run under ``torch.inference_mode``.

    :param velocity: The prior's velocity module, implementing ``get_vf_fn``.
    :param path: The probability path the prior was trained on.
    :param potential: Maps samples of shape ``(N, D)`` to potentials of shape ``(N,)``, row-wise
        independent, e.g. ``lambda x: surrogate({"dose": x})`` for a :class:`SurrogatePotential`.
    :param scale: Multiplies $\lambda_t$; ``0`` samples the prior without evaluating the potential.
    :param max_strength: Upper bound on $b_t$, which diverges as $\beta_t \to 0$.
    """

    velocity: NeuralVelocity
    potential: Callable[[torch.Tensor], torch.Tensor]

    def __init__(
        self,
        velocity: NeuralVelocity,
        path: BasePath,
        potential: Callable[[torch.Tensor], torch.Tensor],
        *,
        scale: float = 1.0,
        max_strength: float = 10.0,
    ) -> None:
        super().__init__()
        self.velocity, self.path, self.potential = velocity, path, potential
        self.scale, self.max_strength = scale, max_strength

    def get_vf_fn(
        self,
        condition_dict: dict[str, torch.Tensor] | None = None,
        source: torch.Tensor | None = None,
    ) -> VfFn:
        vf = self.velocity.get_vf_fn(condition_dict=condition_dict, source=source)
        if self.scale == 0:
            return vf

        def guided(t: torch.Tensor, xt: torch.Tensor) -> torch.Tensor:
            if torch.is_inference_mode_enabled():
                raise RuntimeError("Loss guidance needs autograd; call it outside `torch.inference_mode`.")
            c_x, c_v, lam = guidance_coefficients(self.path, t)
            with torch.enable_grad():
                x = xt.detach().requires_grad_(True)
                v = vf(t, x)
                (grad,) = torch.autograd.grad(self.potential(c_x * x + c_v * v).sum(), x)
            return v.detach() - (self.scale * lam).clamp(0, self.max_strength) * grad

        return guided
