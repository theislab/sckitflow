from typing import Any

import torch

from sc_flow.backends.torch.methods._base_methods import BaseGenerativeFlow
from sc_flow.backends.torch.nn._vf import BaseVelocityField, MLPVelocity

__all__ = ["FlowMatching"]


class FlowMatching(BaseGenerativeFlow):
    _module_cls: type[BaseVelocityField] = MLPVelocity

    def _compute_loss(
        self,
        latent: torch.Tensor,
        source: torch.Tensor | None,
        target: torch.Tensor | None,
        condition_data: dict[str, torch.Tensor] | None,
        group_data: dict[str, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        batch_size = latent.shape[0]
        t = self._time_sampler((batch_size,), device=latent.device, dtype=latent.dtype)
        xt = self._probability_path.compute_xt(t, latent, target)
        ut = self._probability_path.compute_ut(t, xt, latent, target)
        cond = {
            **condition_data,
            **group_data,
        }
        vt = self._module(t, xt, condition_dict=cond, source=source)
        loss = torch.nn.functional.mse_loss(vt, ut)

        return loss, {"loss": loss.item()}

    def predict(self, *args, **kwargs):
        raise NotImplementedError
