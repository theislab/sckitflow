"""The one LightningModule that trains any (model, objective) pair.

This is the single training harness the redesign converges on. It holds the torch
``model`` (the weights) and an :class:`~sc_flow.core.training._objective.Objective`,
and its ``training_step`` just asks the objective for a loss. Whether that loss was
computed in torch or in JAX (via the DLPack bridge) is entirely the objective's
concern — the harness, the optimizer, and the Lightning loop are shared. This is what
collapses the previous two LightningModules (``LitSCFlowModule`` and
``CellFlowJaxModule``) into one.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import lightning.pytorch as pl
import torch

from sc_flow.core.training._objective import Objective

__all__ = ["SCFlowLightningModule"]

# Turns a validation batch (one held-out condition) into (pred, target) torch tensors for the metrics.
PredictFn = Callable[[torch.nn.Module, Any], "tuple[torch.Tensor, torch.Tensor]"]


class SCFlowLightningModule(pl.LightningModule):
    """Train a torch ``model`` under an :class:`Objective`, with optional held-out validation.

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
    val_metrics
        ``{name: torchmetrics.Metric}`` accumulated over the held-out split — each validation batch is
        one condition, and every metric ``update(pred, target)`` compares the predicted vs. target cell
        populations for that condition. ``None`` disables validation entirely (the harness then behaves
        exactly like the train-only path). Registered as submodules so Lightning moves them to the device.
    predict_fn
        Maps ``(model, val_batch) -> (pred, target)`` (see :data:`PredictFn`) — the flow-specific
        translation the metrics score. Required when ``val_metrics`` is given.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        objective: Objective,
        *,
        lr: float = 1e-3,
        optimizer_cls: type[torch.optim.Optimizer] = torch.optim.Adam,
        val_metrics: Mapping[str, torch.nn.Module] | None = None,
        predict_fn: PredictFn | None = None,
        val_max_source_cells: int | None = 2048,
        debug_val: bool = False,
    ) -> None:
        super().__init__()
        self.model = model
        self._objective = objective
        self._lr = lr
        self._optimizer_cls = optimizer_cls
        # torchmetrics are nn.Modules — a ModuleDict registers them so Lightning moves them to the device.
        self._val_metrics = torch.nn.ModuleDict(dict(val_metrics)) if val_metrics else None
        # Identity baseline: the SAME metrics scored on the untouched source (control) vs the target — the
        # "predict nothing" reference tahoe_eval.py prints as `id=`. Logged every pass as `<name>_identity`
        # so a bare negative r_squared is readable: model-worse-than-identity is a real regression, whereas
        # identity scoring just as badly means the eval population itself is mismatched (e.g. controls pooled
        # across plates vs a within-plate target), not a model/harness bug. Fresh clones — separate state.
        self._id_metrics = (
            torch.nn.ModuleDict({k: m.clone() for k, m in dict(val_metrics).items()}) if val_metrics else None
        )
        self._predict_fn = predict_fn
        # binded's EvalLoader reads each held-out control population IN FULL (by design — see
        # binded._eval_loader), which can be tens of thousands of cells once match_context pools controls
        # across many plates/stores. Both the ODE trajectory (integrate_translation) and the O(n^2)
        # pairwise-distance metrics (EnergyDistance) scale with that count, so an uncapped population
        # reliably OOMs at real multi-plate scale. Subsample the SOURCE (control) population before
        # predict/scoring; None disables the cap (matches the old, unsafe-at-scale behavior).
        self._val_max_source_cells = val_max_source_cells
        self._debug_val = debug_val
        # {metric_name: [mean-over-conditions per validation pass]} — read back by FlowMatching after fit.
        self.metrics_history: dict[str, list[float]] = {}

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        loss, logs = self._objective.compute_loss(self.model, batch)
        for key, value in logs.items():
            self.log(key, value, prog_bar=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        if self._val_metrics is None or self._predict_fn is None:
            return
        cap = self._val_max_source_cells
        source = batch["source"]
        n_source = source.shape[0]
        if cap is not None and n_source > cap:
            idx = torch.randperm(source.shape[0], device=source.device)[:cap]
            batch = {**batch, "source": source[idx]}
        pred, target = self._predict_fn(self.model, batch)
        # identity baseline = the (subsampled) source fed to predict, untouched, on pred's dtype/device.
        src = batch["source"].to(device=pred.device, dtype=pred.dtype)
        if self._debug_val:
            leaf = batch.get("leaf")
            pm, ps = float(pred.abs().mean()), float(pred.std())
            tm, ts = float(target.abs().mean()), float(target.std())

            def _r2(p: torch.Tensor, t: torch.Tensor) -> float:  # per-condition R² of feature-wise means
                pmean, tmean = p.mean(0), t.mean(0)
                ss_res = ((tmean - pmean) ** 2).sum()
                ss_tot = ((tmean - tmean.mean()) ** 2).sum().clamp_min(1e-12)
                return float(1.0 - ss_res / ss_tot)

            print(f"[val-debug] leaf={leaf} n_source={n_source} n_target={target.shape[0]} "
                  f"pred(mean_abs={pm:.3f} std={ps:.3f}) target(mean_abs={tm:.3f} std={ts:.3f}) "
                  f"r2_model={_r2(pred, target):+.3f} r2_id={_r2(src, target):+.3f}", flush=True)
        for metric in self._val_metrics.values():
            metric.update(pred, target)
        if self._id_metrics is not None:
            for metric in self._id_metrics.values():
                metric.update(src, target)

    def on_validation_epoch_end(self) -> None:
        if self._val_metrics is None:
            return
        for name, metric in self._val_metrics.items():
            value = metric.compute()
            self.log(f"val_{name}_mean", value, prog_bar=True)
            self.metrics_history.setdefault(name, []).append(float(value))
            metric.reset()
        if self._id_metrics is not None:
            for name, metric in self._id_metrics.items():
                self.metrics_history.setdefault(f"{name}_identity", []).append(float(metric.compute()))
                metric.reset()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return self._optimizer_cls(self.model.parameters(), lr=self._lr)
