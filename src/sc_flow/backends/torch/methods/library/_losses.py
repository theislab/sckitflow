from collections.abc import Callable
from typing import Any

import torch

from sc_flow.backends.torch.nn._modules import BaseModule
from sc_flow.backends.torch.probability_paths._probability_paths import BaseProbabilityPath

__all__ = [
    "compute_lmd_loss",
    "compute_emd_loss",
    "compute_ect_loss",
    "compute_fmm_loss",
]


def compute_lmd_loss(
    s: torch.Tensor,
    t: torch.Tensor,
    latent: torch.Tensor,
    cond: dict[str, torch.Tensor],
    target_state: torch.Tensor,
    student_module: BaseModule,
    probability_path: BaseProbabilityPath,
    source_state: torch.Tensor | None,
    teacher_module: BaseModule | None,
    weight_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, Any]]:
    # sample ground truth interpolant
    xs = probability_path.compute_xt(s, latent, target_state)

    # forward pass on neural networks with jvp
    xts_hat, dXdt = torch.func.jvp(
        student_module.get_vf_fn(cond, source=source_state),
        (s, t, xs),
        (torch.zeros_like(s), torch.ones_like(t), torch.zeros_like(xs)),
    )

    # evaluate vf
    vf_fn = teacher_module.get_vf_fn(cond, source=source_state)
    vt = vf_fn(t, xts_hat)

    # compute loss
    loss = torch.mean(weight_fn(s, t) * ((dXdt - vt) ** 2).sum(-1))
    return loss, {"loss": loss.item()}


def compute_emd_loss(
    s: torch.Tensor,
    t: torch.Tensor,
    latent: torch.Tensor,
    cond: dict[str, torch.Tensor],
    target_state: torch.Tensor,
    student_module: BaseModule,
    probability_path: BaseProbabilityPath,
    source_state: torch.Tensor | None,
    teacher_module: BaseModule | None,
    weight_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, Any]]:
    # sample ground truth interpolant
    xs = probability_path.compute_xt(s, latent, target_state)

    # evaluate vf
    vf_fn = teacher_module.get_vf_fn(cond, source=source_state)
    vs = vf_fn(s, xs)

    # compile map function
    map_fn = student_module.get_vf_fn(cond, source=source_state)

    # forward pass on neural networks with jvp and compute time differential
    _, dXds = torch.func.jvp(
        map_fn,
        (s, t, xs),
        (torch.ones_like(s), torch.zeros_like(t), torch.zeros_like(xs)),
    )

    # forward pass on neural networks with jvp and compute time differential
    _, nablaXs_fn = torch.func.vjp(
        lambda xs: map_fn(s, t, xs),
        xs,
    )
    nablaXs = nablaXs_fn(vs)[0]

    # compute loss
    loss = torch.mean(weight_fn(s, t) * ((nablaXs + dXds) ** 2).sum(-1))
    return loss, {"loss": loss.item()}


def compute_ect_loss(
    s: torch.Tensor,
    t: torch.Tensor,
    latent: torch.Tensor,
    cond: dict[str, torch.Tensor],
    target_state: torch.Tensor,
    student_module: BaseModule,
    probability_path: BaseProbabilityPath,
    source_state: torch.Tensor | None,
    teacher_module: BaseModule | None,
    weight_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, Any]]:
    # handle time direction
    s_new = torch.minimum(s, t)
    t_new = torch.maximum(s, t)
    s = s_new
    t = t_new

    # sample ground truth interpolant
    xs = probability_path.compute_xt(s, latent, target_state)
    us = probability_path.compute_ut(s, latent, target_state, xs)

    # compile map function
    map_fn = student_module.get_vf_fn(cond, source=source_state)

    # forward pass on neural networks with jvp
    # for time differential
    _, dXds = torch.func.jvp(
        map_fn,
        (s, t, xs),
        (torch.ones_like(s), torch.zeros_like(t), torch.zeros_like(xs)),
    )

    # forward pass on neural network for
    # spatial gradients
    def _compute_flow(x: torch.Tensor):
        return map_fn(s, t, x)

    with torch.no_grad():
        _, spatial_term = torch.func.jvp(_compute_flow, (xs,), (us,))
        spatial_term = spatial_term.detach()

    # compute losses
    loss = torch.mean(weight_fn(s, t) * ((dXds + spatial_term) ** 2).sum(-1))
    return loss, {"loss": loss.item()}


def compute_fmm_loss(
    s: torch.Tensor,
    t: torch.Tensor,
    latent: torch.Tensor,
    cond: dict[str, torch.Tensor],
    target_state: torch.Tensor,
    student_module: BaseModule,
    probability_path: BaseProbabilityPath,
    source_state: torch.Tensor | None,
    teacher_module: BaseModule | None,
    weight_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, Any]]:
    # sample ground truth interpolant
    xt = probability_path.compute_xt(t, latent, target_state)
    ut = probability_path.compute_ut(t, latent, target_state, xt)

    # flow to s
    xs_hat = student_module(t, s, xt, condition_dict=cond, source=source_state)

    # forward pass on neural networks with jvp
    xts_hat, dXdt = torch.func.jvp(
        student_module.get_vf_fn(cond, source=source_state),
        (s, t, xs_hat),
        (torch.zeros_like(s), torch.ones_like(t), torch.zeros_like(xt)),
    )

    # compute losses
    loss_tang = torch.mean(weight_fn(s, t) * ((dXdt - ut) ** 2).sum(-1))
    loss_cons = torch.mean(weight_fn(s, t) * ((xts_hat - xt) ** 2).sum(-1))
    loss = loss_tang + loss_cons
    return loss, {"loss": loss.item(), "loss_tang": loss_tang.item(), "loss_cons": loss_cons.item()}
