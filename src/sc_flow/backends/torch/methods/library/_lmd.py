from typing import Any

import torch

from sc_flow.backends.torch.methods.library._base import BaseConsistencyModel

__all__ = ["LMD"]


class LMD(BaseConsistencyModel):
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
        xs = self._probability_path.compute_xt(s, latent, target_state)

        # forward pass on neural networks with jvp
        xts_hat, dXdt = torch.func.jvp(
            self._module.get_vf_fn(cond, source=source_state),
            (s, t, xs),
            (torch.zeros_like(s), torch.ones_like(t), torch.zeros_like(xs)),
        )
        # evaluate vf
        vf_fn = self.teacher_vf.get_vf_fn(cond, source=source_state)
        vt = vf_fn(t, xts_hat)
        loss = torch.mean(self._weight_fn(s, t) * ((dXdt - vt) ** 2).sum(-1))
        return loss, {"loss": loss.item()}
