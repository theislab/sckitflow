"""Population-level held-out validation as a Lightning ``Callback``.

:class:`~sc_flow.training.TrainingModule` is training-only by design — a generic trainer must not know a
batch has ``source``/``target``/``condition`` semantics. This callback supplies the distribution-matching
eval protocol that flow matching (and any control→perturbed model) needs, *composed* from the orthogonal
seams rather than baked into the module:

* it drives inference through an injected :class:`~sc_flow.training.Predictor` — the **same** object
  ``FlowMatching.predict`` uses on external data, so a validation metric reflects exactly what inference
  does;
* it scores prediction *streams* against the held-out target — the model, and an **identity baseline**
  (the untouched control, i.e. "predict nothing"). The identity baseline is a perturbation-domain
  reference (meaningful only for control→perturbed matching), which is why it lives here and not in the
  generic trainer. It is logged as ``<metric>_identity`` alongside the model's ``<metric>``.

**Classifier-free guidance sweep.** When ``guidance_predictors`` is given (one :class:`Predictor` per
guidance scale ``w``), the callback scores the *same* held-out batch once per ``w`` and logs every scale
as ``val_<metric>_mean_gs<w>``. It then selects the ``w`` that optimizes the primary metric and surfaces
its numbers under the un-suffixed ``val_<metric>_mean`` (plus ``val_best_guidance_scale``), so graphing,
the sweep objective, and any scale-agnostic downstream read always see the best guided result.

Validation is optional: with no predictor or no metrics the callback is inert, and the facade only attaches
it (and a val dataloader) when both are present.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import lightning.pytorch as pl
import torch

from sc_flow.training import Predictor

__all__ = ["PerturbationValidationCallback"]

#: Prediction streams scored each pass, keyed by the suffix appended to each metric's logged name:
#: ``""`` = the model (no-sweep), ``"_identity"`` = the predict-nothing baseline (untouched control), and
#: ``"_gs<w>"`` = the model guided at scale ``w`` (sweep).
_MODEL, _IDENTITY = "", "_identity"

#: Substrings marking a metric as higher-is-better; anything else is treated as lower-is-better. Used only
#: to pick the winning guidance scale (r_squared/correlation up, e-distance/mse down).
_MAXIMIZE_SUBSTRINGS = ("r_squared", "r2", "corr", "pearson", "spearman", "accuracy")


def _clone_metrics(templates: Mapping[str, torch.nn.Module]) -> dict[str, torch.nn.Module]:
    """A fresh, independent copy of each metric. torchmetrics accumulate state, so every scored stream
    needs its own instances — this is why the identity baseline can't reuse the model's metrics."""
    return {name: metric.clone() for name, metric in templates.items()}


def _higher_is_better(metric_name: str) -> bool:
    name = metric_name.lower()
    return any(tok in name for tok in _MAXIMIZE_SUBSTRINGS)


class PerturbationValidationCallback(pl.Callback):
    """Control→perturbed population validation, attachable to any :class:`~sc_flow.training.TrainingModule`.

    Parameters
    ----------
    predictor
        Pure inference seam ``predict(model, batch) -> pred`` (the flow ODE integrator), shared with
        ``FlowMatching.predict``. ``None`` makes the callback inert. Used as the single model stream when
        no guidance sweep is configured.
    val_metrics
        ``{name: torchmetrics.Metric}`` — used as *templates*: each scored stream gets an independent
        clone. Each validation batch is one condition, each ``update(pred, target)`` compares predicted vs.
        target cell populations. ``None`` makes the callback inert.
    val_max_source_cells
        Cap on the control (``source``) population fed to ``predict``/scoring; ``None`` disables it. The
        held-out control population can be tens of thousands of cells once a match context pools controls
        across stores, so both the ODE trajectory and the O(n^2) pairwise-distance metrics reliably OOM
        uncapped at real scale.
    guidance_predictors
        Optional ``{guidance_scale: Predictor}`` — enables the CFG sweep. Each predictor scores the same
        held-out batch; every scale is logged as ``val_<metric>_mean_gs<w>`` and the best (by
        ``primary_metric``) is surfaced under the un-suffixed keys. ``None``/empty ⇒ no sweep (the single
        ``predictor`` model stream).
    primary_metric
        The metric that selects the winning guidance scale; defaults to the first ``val_metrics`` key.
    """

    def __init__(
        self,
        *,
        predictor: Predictor | None,
        val_metrics: Mapping[str, torch.nn.Module] | None,
        val_max_source_cells: int | None = 2048,
        guidance_predictors: Mapping[float, Predictor] | None = None,
        primary_metric: str | None = None,
    ) -> None:
        super().__init__()
        self._val_max_source_cells = val_max_source_cells
        templates = dict(val_metrics) if val_metrics else {}
        self._active = predictor is not None and bool(templates)

        # Model-prediction streams keyed by the suffix appended to each metric's logged name. Without a
        # sweep that is the single model stream (""); with a sweep it is one stream per scale ("_gs<w>").
        self._suffix_to_w: dict[str, float] = {}
        sweep = dict(guidance_predictors) if guidance_predictors else {}
        if self._active and sweep:
            self._predictors: dict[str, Predictor] = {}
            for w, pred in sweep.items():
                suffix = f"_gs{float(w):g}"
                self._predictors[suffix] = pred
                self._suffix_to_w[suffix] = float(w)
        else:
            self._predictors = {_MODEL: predictor} if self._active else {}
        self._sweeping = bool(self._suffix_to_w)
        self._primary_metric = primary_metric or (next(iter(templates)) if templates else None)

        # One independent metric set per scored stream (see _clone_metrics): each prediction stream + the
        # identity baseline.
        stream_suffixes = [*self._predictors, *([_IDENTITY] if self._active else [])]
        self._stream_metrics: dict[str, dict[str, torch.nn.Module]] = (
            {suffix: _clone_metrics(templates) for suffix in stream_suffixes} if self._active else {}
        )
        # {metric_name[suffix]: [mean-over-conditions per validation pass]} — read back by FlowMatching.
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
        # Every model-prediction stream (one, or one per guidance scale) + the identity "predict-nothing"
        # baseline (the capped control, untouched), all scored against the same target.
        predictions = {suffix: pred.predict(pl_module.model, batch) for suffix, pred in self._predictors.items()}
        predictions[_IDENTITY] = batch["source"]
        for suffix, pred in predictions.items():
            for metric in self._stream_metrics[suffix].values():
                metric.to(pred.device)  # metrics live on the callback; place state on-device (idempotent)
                metric.update(pred, target)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self._active:
            return
        computed: dict[str, dict[str, float]] = {}
        for suffix, metrics in self._stream_metrics.items():
            values: dict[str, float] = {}
            for name, metric in metrics.items():
                values[name] = float(metric.compute())
                metric.reset()
            computed[suffix] = values

        if getattr(trainer, "sanity_checking", False):
            return  # startup sanity pass: exercise the whole held-out path, but don't record init

        # Log every scored stream so wandb shows the model (or every guidance scale) AND the identity
        # baseline side by side (val_<name>_mean[_gs<w>|_identity]); prog bar stays model-only.
        for suffix, values in computed.items():
            for name, value in values.items():
                pl_module.log(f"val_{name}_mean{suffix}", value, prog_bar=suffix == _MODEL)
                self.metrics_history.setdefault(f"{name}{suffix}", []).append(value)

        # CFG sweep: surface the best scale under the un-suffixed keys (so the sweep objective + graphs are
        # scale-agnostic) and record which scale won.
        if self._sweeping and self._primary_metric is not None:
            best_suffix = self._select_best_suffix(computed)
            best_w = self._suffix_to_w[best_suffix]
            for name, value in computed[best_suffix].items():
                pl_module.log(f"val_{name}_mean", value, prog_bar=True)
                self.metrics_history.setdefault(name, []).append(value)
            pl_module.log("val_best_guidance_scale", best_w, prog_bar=True)
            self.metrics_history.setdefault("best_guidance_scale", []).append(best_w)

    def _select_best_suffix(self, computed: dict[str, dict[str, float]]) -> str:
        """The guidance-scale stream that optimizes the primary metric (max if higher-is-better, else min)."""
        metric = self._primary_metric
        chooser = max if _higher_is_better(metric) else min
        return chooser(self._suffix_to_w, key=lambda suffix: computed[suffix][metric])

    def _cap_source(self, batch: Any) -> Any:
        cap = self._val_max_source_cells
        source = batch["source"]
        if cap is not None and source.shape[0] > cap:
            idx = torch.randperm(source.shape[0], device=source.device)[:cap]
            return {**batch, "source": source[idx]}
        return batch
