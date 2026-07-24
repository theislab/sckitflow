"""The one Lightning module that trains any ``(model, objective)`` pair.

Deliberately model-family-agnostic: it holds a torch ``model`` (the weights) and an
:class:`~sc_flow.training._objective.Objective`, and its ``training_step`` just asks the objective for a
loss. Whether that loss was computed in torch or in JAX (via a DLPack bridge) is entirely the objective's
concern — the module, optimizer, and Lightning loop are shared.

It defines **no validation logic**. A generic trainer must not assume a batch carries ``source``/
``target``/``condition`` semantics, so the held-out / population-level scoring protocol is supplied
downstream by attaching a Lightning ``Callback`` (e.g. ``sc_flow.flow.PerturbationValidationCallback``).
Keeping the eval protocol out of the base is what lets a foundation-model toolbox reuse the same trainer
without inheriting perturbation eval.
"""

from __future__ import annotations

from collections.abc import Collection
from typing import Any

import lightning.pytorch as pl
import torch

from sc_flow.training._objective import Objective

__all__ = ["TrainingModule"]


class TrainingModule(pl.LightningModule):
    """Train a torch ``model`` under an :class:`Objective`. Training-only; see the module docstring.

    Parameters
    ----------
    model
        The torch ``nn.Module`` holding the weights. The optimizer steps ``model.parameters()`` — the
        single source of truth even when the objective computes in JAX.
    objective
        Turns a batch into ``(loss, logs)`` (see :class:`Objective`).
    lr
        Learning rate for the default optimizer.
    optimizer_cls
        Optimizer class constructed over ``model.parameters()``.
    prog_bar_metrics
        Which objective log keys to surface on the progress bar. ``None`` (default) shows every key the
        objective returns; pass a collection to restrict the bar to those keys — the rest are still logged
        (``prog_bar=False``), just not displayed.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        objective: Objective,
        *,
        lr: float = 1e-3,
        optimizer_cls: type[torch.optim.Optimizer] = torch.optim.Adam,
        prog_bar_metrics: Collection[str] | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self._objective = objective
        self._lr = lr
        self._optimizer_cls = optimizer_cls
        self._prog_bar_metrics = None if prog_bar_metrics is None else frozenset(prog_bar_metrics)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        loss, logs = self._objective.compute_loss(self.model, batch)
        for key, value in logs.items():
            on_bar = self._prog_bar_metrics is None or key in self._prog_bar_metrics
            self.log(key, value, prog_bar=on_bar)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        """No-op — the scoring protocol lives in a Lightning ``Callback``, not here.

        Lightning only runs its validation loop (and fires the ``on_validation_*`` callback hooks) when the
        module defines ``validation_step``. This stub exists solely to open that loop; it deliberately reads
        nothing from ``batch`` so the generic trainer stays free of any ``source``/``target``/``condition``
        vocabulary. A downstream toolbox supplies the actual eval by attaching a ``Callback`` (e.g.
        ``sc_flow.flow.PerturbationValidationCallback``).
        """
        return None

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return self._optimizer_cls(self.model.parameters(), lr=self._lr)
