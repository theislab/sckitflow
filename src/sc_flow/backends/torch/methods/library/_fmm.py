from typing import Any

import torch

from sc_flow.backends.torch.methods.library._base import BaseConsistencyModel

__all__ = ["FMM"]


class FMM(BaseConsistencyModel):
    def _compute_loss(
        self,
        s: torch.Tensor,
        t: torch.Tensor,
        latent: torch.Tensor,
        cond: dict[str, torch.Tensor],
        target_state: torch.Tensor,
        source_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        # sample ground truth interpolant
        xt = self._probability_path.compute_xt(t, latent, target_state)
        ut = self._probability_path.compute_ut(t, latent, target_state, xt)

        # flow to s
        xs_hat = self._module(t, s, xt, condition_dict=cond, source=source_state)

        # forward pass on neural networks with jvp
        xts_hat, dXdt = torch.func.jvp(
            self._module.get_vf_fn(cond, source=source_state),
            (s, t, xs_hat),
            (torch.zeros_like(s), torch.ones_like(t), torch.zeros_like(xt)),
        )

        # compute losses
        loss_tang = torch.mean(self._weight_fn(s, t) * ((dXdt - ut) ** 2).sum(-1))
        loss_cons = torch.mean(self._weight_fn(s, t) * ((xts_hat - xt) ** 2).sum(-1))
        loss = loss_tang + loss_cons
        return loss, {"loss": loss.item(), "loss_cons": loss_cons.item(), "loss_tang": loss_tang.item()}
