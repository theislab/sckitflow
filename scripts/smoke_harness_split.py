"""Smoke: the scfit generic trainer vs. the sc_flow perturbation-validation callback.

Verifies the core/flow split of the training module:
  * ``scfit.training.TrainingModule`` is training-only — no validation logic, no source/target/metric
    vocabulary — and its ``prog_bar_metrics`` restricts which objective logs hit the progress bar.
  * ``sc_flow.flow.PerturbationValidationCallback`` supplies control->perturbed population validation on
    top, driven by an injected ``Predictor``, populating ``metrics_history`` (model + identity baseline).
  * Validation is optional: no callback / no predictor => no scoring.

No data pipeline / jax needed — dummy model, objective, predictor, and tensors only.

    python scripts/smoke_harness_split.py
"""

from __future__ import annotations

import lightning.pytorch as pl
import torch

from scfit.metrics import MeanAggregatedRSquared, PredictionDispersion
from scfit.training import Objective, Predictor, TrainingModule
from sc_flow.flow import ODEPredictor, PerturbationValidationCallback

D = 4


class _DummyVF(torch.nn.Module):
    """Matches the velocity-field forward signature integrate_translation calls."""

    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(D, D)

    def forward(self, t, y, cond_t=None, source_cells=None, condition_mask=None):
        return self.lin(y)


class _DummyObjective(Objective):
    def compute_loss(self, model, batch):
        out = model(batch["x"])
        loss = (out**2).mean()
        return loss, {"loss": loss.detach(), "aux": torch.tensor(1.0)}


class _DummyPredictor(Predictor):
    def predict(self, model, batch):
        return model(batch["source"])


class _TrainDS(torch.utils.data.IterableDataset):
    def __iter__(self):
        for _ in range(4):
            yield {"x": torch.randn(8, D)}


class _ValDS(torch.utils.data.IterableDataset):
    def __iter__(self):
        for _ in range(2):
            yield {"source": torch.randn(6, D), "target": torch.randn(5, D) + 1.0, "leaf": "cond"}


def _trainer(callbacks=None, **kw):
    return pl.Trainer(
        logger=False, enable_checkpointing=False, enable_progress_bar=False,
        enable_model_summary=False, accelerator="cpu", callbacks=callbacks, **kw,
    )


def _train_loader():
    return torch.utils.data.DataLoader(_TrainDS(), batch_size=None)


def main() -> None:
    # 1. Generic base: trains, is validation-free, and never subclassed.
    assert "validation_step" in TrainingModule.__dict__, "base needs a stub validation_step for the loop"
    base = TrainingModule(torch.nn.Linear(D, D), _DummyObjective(), lr=1e-3, prog_bar_metrics={"loss"})
    assert base._prog_bar_metrics == frozenset({"loss"}), "prog_bar_metrics must be honored"
    _trainer(max_steps=4).fit(base, _train_loader())
    print("[ok] scfit.TrainingModule trained (no validation logic; prog_bar_metrics honored)")

    # 2. Optional validation: base + a val loader but NO callback => no scoring, no error.
    _trainer(max_steps=4, val_check_interval=2, num_sanity_val_steps=0, limit_val_batches=2).fit(
        TrainingModule(torch.nn.Linear(D, D), _DummyObjective(), lr=1e-3),
        _train_loader(), torch.utils.data.DataLoader(_ValDS(), batch_size=None),
    )
    print("[ok] validation is optional (no callback => inert, no crash)")

    # 3. Callback supplies population validation + identity baseline (predictor injected).
    cb = PerturbationValidationCallback(
        predictor=_DummyPredictor(),
        val_metrics={"r2": MeanAggregatedRSquared(), "pred_std": PredictionDispersion("std")},
        val_max_source_cells=4,
    )
    _trainer(callbacks=[cb], max_steps=4, val_check_interval=2, num_sanity_val_steps=0, limit_val_batches=2).fit(
        TrainingModule(torch.nn.Linear(D, D), _DummyObjective(), lr=1e-3),
        _train_loader(), torch.utils.data.DataLoader(_ValDS(), batch_size=None),
    )
    assert "r2" in cb.metrics_history, cb.metrics_history
    assert "r2_identity" in cb.metrics_history, "identity baseline must be logged"
    # monitor metric flows through the same stream machinery, model vs identity
    assert "pred_std" in cb.metrics_history and "pred_std_identity" in cb.metrics_history, cb.metrics_history
    assert len(cb.metrics_history["r2"]) >= 1
    print(f"[ok] PerturbationValidationCallback validated (incl. monitor metric): {cb.metrics_history}")

    # 4. Inert callback: no predictor => no history.
    inert = PerturbationValidationCallback(predictor=None, val_metrics={"r2": MeanAggregatedRSquared()})
    _trainer(callbacks=[inert], max_steps=4, val_check_interval=2, num_sanity_val_steps=0, limit_val_batches=2).fit(
        TrainingModule(torch.nn.Linear(D, D), _DummyObjective(), lr=1e-3),
        _train_loader(), torch.utils.data.DataLoader(_ValDS(), batch_size=None),
    )
    assert inert.metrics_history == {}, "callback with no predictor must stay inert"
    print("[ok] callback with no predictor stays inert")

    # 5. Predictor aux path: same inference engine, optional trajectory/stats.
    vf = _DummyVF()
    predictor = ODEPredictor(is_genot=False, state_dim=D, num_steps=5, seed=0)
    batch = {"source": torch.randn(7, D)}
    pred = predictor.predict(vf, batch)
    pred_aux, aux = predictor.predict_with_aux(vf, batch)
    assert pred.shape == (7, D)
    assert torch.allclose(pred, pred_aux), "predict and predict_with_aux must agree on the prediction"
    assert aux["trajectory"].shape == (5, 7, D), aux["trajectory"].shape
    print(f"[ok] Predictor.predict_with_aux returns trajectory {tuple(aux['trajectory'].shape)}")

    print("\nSMOKE PASSED")


if __name__ == "__main__":
    main()
