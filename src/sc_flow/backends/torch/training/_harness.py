"""The one LightningModule that trains any (model, objective) pair.

This is the single training harness the redesign converges on. It holds the torch
``model`` (the weights) and an :class:`~sc_flow.backends.torch.training._objective.Objective`,
and its ``training_step`` just asks the objective for a loss. Whether that loss was
computed in torch or in JAX (via the DLPack bridge) is entirely the objective's
concern — the harness, the optimizer, and the Lightning loop are shared. This is what
collapses the previous two LightningModules (``LitSCFlowModule`` and
``CellFlowJaxModule``) into one.
"""

from __future__ import annotations

from typing import Any

import lightning.pytorch as pl
import torch

from sc_flow.backends.torch.training._objective import Objective

__all__ = ["SCFlowLightningModule"]


class SCFlowLightningModule(pl.LightningModule):
    """Train a torch ``model`` under an :class:`Objective`.

    Parameters
    ----------
    model
        The torch ``nn.Module`` holding the weights (the registerable architecture).
        The optimizer steps ``model.parameters()``; these are the single source of
        truth even when the objective computes in JAX.
    objective
        Turns a batch into ``(loss, logs)`` (see :class:`Objective`).
    lr
        Learning rate for the default optimizer.
    optimizer_cls
        Optimizer class constructed over ``model.parameters()``.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        objective: Objective,
        *,
        lr: float = 1e-3,
        optimizer_cls: type[torch.optim.Optimizer] = torch.optim.Adam,
    ) -> None:
        super().__init__()
        self.model = model
        self._objective = objective
        self._lr = lr
        self._optimizer_cls = optimizer_cls

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        loss, logs = self._objective.compute_loss(self.model, batch)
        for key, value in logs.items():
            self.log(key, value, prog_bar=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return self._optimizer_cls(self.model.parameters(), lr=self._lr)
