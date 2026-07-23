"""Population-level held-out validation as a Lightning ``Callback``.

:class:`~scfit.training.TrainingModule` is training-only by design — a generic trainer must not know a
batch has ``source``/``target``/``condition`` semantics. This callback supplies the distribution-matching
eval protocol that flow matching (and any control→perturbed model) needs, *composed* from the orthogonal
seams rather than baked into the module:

* it drives inference through an injected :class:`~scfit.training.Predictor` — the **same** object
  ``FlowMatching.predict`` uses on external data, so a validation metric reflects exactly what inference
  does;
* it scores two prediction *streams* against the held-out target — the model, and an **identity
  baseline** (the untouched control, i.e. "predict nothing"). The identity baseline is a
  perturbation-domain reference (meaningful only for control→perturbed matching), which is why it lives
  here and not in the generic trainer. It is logged as ``<metric>_identity`` alongside the model's
  ``<metric>``.

Validation is optional: with no predictor or no metrics the callback is inert, and the facade only attaches
it (and a val dataloader) when both are present.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import lightning.pytorch as pl
import torch

from scfit.training import Predictor

__all__ = ["PerturbationValidationCallback"]

#: Prediction streams scored each pass, keyed by the suffix appended to each metric's logged name:
#: ``""`` = the model, ``"_identity"`` = the predict-nothing baseline (untouched control).
_MODEL, _IDENTITY = "", "_identity"


def _clone_metrics(templates: Mapping[str, torch.nn.Module]) -> dict[str, torch.nn.Module]:
    """A fresh, independent copy of each metric. torchmetrics accumulate state, so every scored stream
    needs its own instances — this is why the identity baseline can't reuse the model's metrics."""
    return {name: metric.clone() for name, metric in templates.items()}


class PerturbationValidationCallback(pl.Callback):
    """Control→perturbed population validation, attachable to any :class:`~scfit.training.TrainingModule`.

    Parameters
    ----------
    predictor
        Pure inference seam ``predict(model, batch) -> pred`` (the flow ODE integrator), shared with
        ``FlowMatching.predict``. ``None`` makes the callback inert.
    val_metrics
        ``{name: torchmetrics.Metric}`` — used as *templates*: each scored stream (model, identity) gets an
        independent clone. Each validation batch is one condition, each ``update(pred, target)`` compares
        predicted vs. target cell populations. ``None`` makes the callback inert.
    val_max_source_cells
        Cap on the control (``source``) population fed to ``predict``/scoring; ``None`` disables it. The
        held-out control population is read in full by the eval loader and can be tens of thousands of
        cells once a match context pools controls across stores, so both the ODE trajectory and the
        O(n^2) pairwise-distance metrics reliably OOM uncapped at real scale.
    """

    def __init__(
        self,
        *,
        predictor: Predictor | None,
        val_metrics: Mapping[str, torch.nn.Module] | None,
        val_max_source_cells: int | None = 2048,
    ) -> None:
        super().__init__()
        self._predictor = predictor
        self._val_max_source_cells = val_max_source_cells
        self._active = predictor is not None and bool(val_metrics)
        templates = dict(val_metrics) if val_metrics else {}
        # One independent metric set per scored stream (see _clone_metrics for why cloning is required).
        self._stream_metrics: dict[str, dict[str, torch.nn.Module]] = (
            {_MODEL: _clone_metrics(templates), _IDENTITY: _clone_metrics(templates)} if self._active else {}
        )
        # {metric_name[_identity]: [mean-over-conditions per validation pass]} — read back by FlowMatching.
        self.metrics_history: dict[str, list[float]] = {}

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if not self._active:
            return
        batch = self._cap_source(batch)
        # source/target are already torch tensors on the model device (loader is to="torch" + Lightning
        # transfers the batch), so no dtype/device coercion — a mismatch should surface, not be papered over.
        target = batch["target"]
        # The two streams scored against the same target: the model, and the identity "predict-nothing"
        # baseline (the capped control, untouched).
        predictions = {_MODEL: self._predictor.predict(pl_module.model, batch), _IDENTITY: batch["source"]}
        for suffix, pred in predictions.items():
            for metric in self._stream_metrics[suffix].values():
                metric.to(pred.device)  # metrics live on the callback; place state on-device (idempotent)
                metric.update(pred, target)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self._active:
            return
        for suffix, metrics in self._stream_metrics.items():
            for name, metric in metrics.items():
                value = float(metric.compute())
                if suffix == _MODEL:
                    pl_module.log(f"val_{name}_mean", value, prog_bar=True)
                self.metrics_history.setdefault(f"{name}{suffix}", []).append(value)
                metric.reset()

    def _cap_source(self, batch: Any) -> Any:
        cap = self._val_max_source_cells
        source = batch["source"]
        if cap is not None and source.shape[0] > cap:
            idx = torch.randperm(source.shape[0], device=source.device)[:cap]
            return {**batch, "source": source[idx]}
        return batch
