from typing import Any
import torch
import sc_flow.methods as basemethods
from sc_flow import _constants, _types

__all__ = ["FlowMatching", "OTFlowMatching", "GENOT"]


class FlowMatching(basemethods.FlowMatching):
    """TODO."""


    def step_fn(
        self,
        *args: Any,
        batch: dict[str, _types.TensorLike],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Training step for flow matching.

        Basically, it gets source and target states, plus the condition,
        and then computes the velocity field and the loss.
        """
        # Use mixed precision context
        with torch.amp.autocast("cuda", enabled=self.use_amp):
            source = batch[_constants.SOURCE_STATE]
            target = batch[_constants.TARGET_STATE]

            if not isinstance(source, torch.torch.Tensor):
                source = torch.from_numpy(source).to(self.device, dtype=torch.float16, non_blocking=True)
            if not isinstance(target, torch.torch.Tensor):
                target = torch.from_numpy(target).to(self.device, dtype=torch.float16, non_blocking=True)

            if self.generate_from_noise:
                latent = torch.randn_like(source, dtype=source.dtype)
            else:
                latent = source

            condition = {}
            for cond, cond_data in batch[_constants.CONDITION_KEY].items():
                if not isinstance(cond_data, torch.torch.Tensor):
                    condition[cond] = torch.from_numpy(cond_data).to(
                        self.device, dtype=torch.float16, non_blocking=True
                    )
                else:
                    condition[cond] = cond_data.to(self.device, dtype=torch.float16, non_blocking=True)

            batch_size = target.shape[0]
            t = self.time_sampler((batch_size,), device=target.device)
            xt = self.flow.compute_x_t(t, latent, target)
            ut = self.flow.compute_u_t(t, latent, target, xt)
            vt = self.velocity_field(t, xt, condition, source=source)
            loss = torch.nn.functional.mse_loss(vt, ut)
            return loss, {"loss": loss.detach().cpu().item()}

    pass


class OTFlowMatching(basemethods.OTFlowMatching):
    """TODO."""


class GENOT(basemethods.GENOT):
    """TODO."""

    pass
