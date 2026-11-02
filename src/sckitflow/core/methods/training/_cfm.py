from __future__ import annotations

from typing import Any

import torch
from scfit.registry import component

from sckitflow.core._data_utils import (
    get_tensor_dict_from_data,
    prepare_latent_train,
)
from sckitflow.core._types import StepData
from sckitflow.core.methods._base import AbstractFlowMethod, FlowMethodConfig, TrainingMethodConfig
from sckitflow.data._dims import DataDimensions

__all__ = ["CFMTraining", "CFMTrainingConfig"]


class CFMTraining(AbstractFlowMethod):
    """Conditional Flow Matching training method.

    Constructed from the module, the flow configuration, and the data shape spec:

    .. code-block:: python

        method = CFMTraining(
            module,
            CFMTrainingConfig(probability_path=...),
            dims=dm.get_data_dimensionalities(adata),
        )
    """

    def __init__(
        self,
        module: torch.nn.Module,
        config: CFMTrainingConfig | None = None,
        *,
        dims: DataDimensions,
    ) -> None:
        """Initializes CFM training.

        :param config: The flow settings; ``None`` takes every default.
        :param dims: The shape spec of the data the module was built for. Its
            ``state_dim`` sizes the training latent.
        """
        self.config = CFMTrainingConfig() if config is None else config
        super().__init__(module, self.config, dims=dims)

    def compute_loss(self, step_data: StepData, *, generator: torch.Generator) -> tuple[torch.Tensor, dict[str, Any]]:
        """The flow-matching loss. Noise, time and path samples are drawn from ``generator``."""
        # ---- Get source and target states from step data ----
        target = step_data["target_state"]
        source = step_data["source_state"]

        # ---- Get conditioning data from step data ----
        cond = {
            **get_tensor_dict_from_data(step_data["target_condition_data"]),
            **get_tensor_dict_from_data(step_data["target_group_data"]),
        }

        # ---- Sample latent (noise) -- shape (B, dims.state_dim) ----
        # Row count comes from `source` when present, else from `cond`. Device and
        # dtype come from the module, which is what will consume the tensor. The
        # target is not read for placement: its shape is the same B either way, and
        # the width belongs to `dims`.
        latent = prepare_latent_train(
            source,
            self.noise_sampler,
            self.dims,
            batch_size=target.shape[0],
            device=self.device,
            dtype=self.dtype,
            generate_from_noise=self.generate_from_noise,
            generator=generator,
        )
        batch_size = latent.shape[0]

        # ---- Sample time ----
        t = self.time_sampler((batch_size,), generator=generator, device=latent.device, dtype=latent.dtype)

        # ---- Sample from probability path: interpolant and velocity ----
        xt = self.probability_path.compute_xt(t, latent, target, generator=generator)
        ut = self.probability_path.compute_ut(t, xt, latent, target)

        # ---- Predict velocity field and compute loss ----
        vt = self.module(t, xt, condition_dict=cond, source=source)
        loss = torch.nn.functional.mse_loss(vt, ut)
        return loss, {"loss": loss.detach()}  # a tensor: Lightning reads it back per logging interval, not per step


@component("training_method.cfm", builds=CFMTraining)
class CFMTrainingConfig(FlowMethodConfig, TrainingMethodConfig):
    """Conditional Flow Matching training."""

    def build(self, module: torch.nn.Module, *, dims: DataDimensions) -> CFMTraining:
        return CFMTraining(module, self, dims=dims)
