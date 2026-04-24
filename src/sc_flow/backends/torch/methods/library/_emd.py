from typing import Any

import torch

from sc_flow.backends.torch.methods._base import TorchGenerativeFlow

__all__ = ["EMD"]


class EMD(TorchGenerativeFlow):
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

        # evaluate vf
        vf_fn = self.teacher_vf.get_vf_fn(cond, source=source_state)
        vs = vf_fn(s, xs)

        # forward pass on neural networks with jvp and compute time differential
        _, dXds = torch.func.jvp(
            self._module.get_vf_fn(cond, source=source_state),
            (s, t, xs),
            (torch.ones_like(s), torch.zeros_like(t), torch.zeros_like(xs)),
        )

        # forward pass on neural networks with jvp and compute time differential
        _, nablaXs_fn = torch.func.vjp(
            lambda xs: self._module.get_vf_fn(cond, source=source_state)(s, t, xs),
            xs,
        )
        nablaXs = nablaXs_fn(vs)[0]

        # compute loss
        loss = torch.mean(self._weight_fn(s, t) * ((nablaXs + dXds) ** 2).sum(-1))
        return loss, {"loss": loss.item()}
