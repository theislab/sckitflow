from typing import Any

import torch

from sckitflow.core._data_utils import (
    get_tensor_dict_from_data,
    prepare_latent_train,
)
from sckitflow.core._types import StepData
from sckitflow.core.methods._base import BaseFlowTrainingProtocol

__all__ = ["CFMTrainingProtocol"]


class CFMTrainingProtocol(BaseFlowTrainingProtocol):
    """Training protocol for Conditional Flow Matching."""

    def compute_loss(self, step_data: StepData) -> tuple[torch.Tensor, dict[str, Any]]:
        # ---- Get source and target states from step data ----
        target = step_data["target_state"].to(device=self.device_id, dtype=self.dtype)
        source_raw = step_data["source_state"]
        source = source_raw.to(device=self.device_id, dtype=self.dtype) if source_raw is not None else None

        # ---- Get conditioning data from step data ----
        condition_data = get_tensor_dict_from_data(step_data["target_condition_data"])
        group_data = get_tensor_dict_from_data(step_data["target_group_data"])
        cond = {
            **{k: v.to(device=self.device_id, dtype=self.dtype) for k, v in condition_data.items()},
            **{k: v.to(device=self.device_id, dtype=self.dtype) for k, v in group_data.items()},
        }

        # ---- Sample latent (noise) – shape (batch_size, dim) -----
        latent = prepare_latent_train(
            source,
            target,
            self.noise_sampler,
            generate_from_noise=self.generate_from_noise,
        ).to(device=self.device_id, dtype=self.dtype)
        batch_size = latent.shape[0]

        # ---- Sample time ----
        t = self.time_sampler((batch_size,), device=latent.device, dtype=latent.dtype)

        # ---- Sample from probability path: interpolant and velocity ----
        xt = self.probability_path.compute_xt(t, latent, target)
        ut = self.probability_path.compute_ut(t, xt, latent, target)

        # ---- Predict velocity field and compute loss ----
        vt = self.module(t, xt, condition_dict=cond, source=source)
        loss = torch.nn.functional.mse_loss(vt, ut)
        return loss, {"loss": loss.item()}
