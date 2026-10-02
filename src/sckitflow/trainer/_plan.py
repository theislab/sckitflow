"""The `lightning.pytorch` module that runs a sckitflow training method.

Everything a training run needs -- the loop, the backward pass, optimizer
stepping, the progress bar, device placement, validation cadence, logging,
checkpointing -- comes from Lightning. This file only says what one step is.
"""

import copy
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import lightning.pytorch as pl
import torch

from sckitflow._predict import prediction_record
from sckitflow._random import generators
from sckitflow.core._types import StepData
from sckitflow.core.methods._base import SupportsInference, SupportsTraining

__all__ = ["TrainingPlan"]

# Stream keys: each stage draws from its own generators, never from another stage's.
_TRAIN, _VALIDATE, _PREDICT = 0, 1, 2


class TrainingPlan(pl.LightningModule):
    """Runs a `SupportsTraining` method, and optionally scores a `SupportsInference` one.

    Hand it to a ``lightning.Trainer``:

    .. code-block:: python

        plan = TrainingPlan(CFMTraining(module=module), torch.optim.Adam(module.parameters()))
        pl.Trainer(max_steps=1000).fit(plan, train_loader)

    :param training_method: The method whose `compute_loss` is one training step.
    :param optimizer: An already-built optimizer over the method's module.
        Taken rather than configured, so there is no optimizer factory here.
        May be `None` for a plan used only to predict, which never asks for one.
    :param lr_scheduler: Optional scheduler for that optimizer.
    :param lr_scheduler_interval: ``"step"`` or ``"epoch"``; how often Lightning
        steps the scheduler.
    :param inference_method: The method used for validation. When `None`, or
        when no metrics are given, validation produces nothing.
    :param metrics: ``{name: torchmetrics.Metric}`` scored on every validation
        batch, with one copy per validation set. Registered as submodules, so
        Lightning moves them.
    :param val_names: Names of the validation sets, positionally matching the
        ``val_dataloaders`` handed to ``lightning.Trainer``. Lightning
        identifies those by index; metrics are logged under a name.
    :param pred_transform: Optional callable applied to predictions before scoring.
    :param target_transform: Optional callable applied to targets before scoring.
    :param predict_kwargs: Forwarded to the inference method's ``predict``.
    :param seed: Every random draw of training, validation and prediction comes from generators derived
        from it and the step, so nothing reads the global RNG.
    """

    def __init__(
        self,
        training_method: SupportsTraining,
        optimizer: torch.optim.Optimizer | None = None,
        *,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        lr_scheduler_interval: str = "step",
        inference_method: SupportsInference | None = None,
        metrics: Mapping[str, torch.nn.Module] | None = None,
        val_names: Sequence[str] = ("val",),
        pred_transform: Callable[[Any], Any] | None = None,
        target_transform: Callable[[Any], Any] | None = None,
        predict_kwargs: dict[str, Any] | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.training_method = training_method
        self.inference_method = inference_method
        # Registers the method's parameters with Lightning, which is what makes
        # optimizer wiring, device placement and train/eval mode work.
        self.module = training_method.module
        self._val_names = list(val_names)
        # one copy per validation set, so sets are scored apart; `ModuleDict` so they ride along to the accelerator
        self.metrics = (
            torch.nn.ModuleDict(
                {val: torch.nn.ModuleDict({k: copy.deepcopy(m) for k, m in metrics.items()}) for val in self._val_names}
            )
            if metrics
            else None
        )
        self._scored: set[str] = set()  # validation sets that saw a batch this epoch

        self._optimizer = optimizer
        self._lr_scheduler = lr_scheduler
        self._lr_scheduler_interval = lr_scheduler_interval
        self._pred_transform = pred_transform
        self._target_transform = target_transform
        self._predict_kwargs = {} if predict_kwargs is None else predict_kwargs
        self.seed = seed

    def training_step(self, batch: StepData, batch_idx: int) -> torch.Tensor:
        generator, rng = generators(self.seed, _TRAIN, self.global_step, device=self.device)
        loss, metrics = self.training_method.compute_loss(batch, generator=generator, rng=rng)
        self.log_dict(metrics, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch: StepData, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Scores one validation batch. Metrics aggregate until the epoch ends."""
        if self.inference_method is None or self.metrics is None:
            return

        generator, _ = generators(self.seed, _VALIDATE, dataloader_idx, batch_idx, device=self.device)
        preds = self.inference_method.predict(batch, generator=generator, **self._predict_kwargs)
        preds = getattr(preds, "X", preds)
        targets = batch["target_state"]
        if self._pred_transform is not None:
            preds = self._pred_transform(preds)
        if self._target_transform is not None:
            targets = self._target_transform(targets)

        val_name = self._val_names[dataloader_idx]
        for metric in self.metrics[val_name].values():
            metric.update(torch.as_tensor(preds), torch.as_tensor(targets))
        self._scored.add(val_name)

    def on_validation_epoch_end(self) -> None:
        scored, self._scored = self._scored, set()
        for val_name in scored:  # empty unless there are metrics and an inference method
            for name, metric in self.metrics[val_name].items():  # pyright: ignore[reportOptionalSubscript]
                if not self.trainer.sanity_checking:
                    self.log(f"{val_name}/{name}", metric.compute())
                metric.reset()

    def predict_step(self, batch: tuple[StepData, tuple], batch_idx: int, dataloader_idx: int = 0) -> dict[str, Any]:
        """Predicts one group and shapes it for reassembly.

        The predict loader yields ``(step_data, leaf)``; ``leaf`` is the group's
        ``group_by`` value tuple, which is what the output ``obs`` is rebuilt
        from. The per-batch obsm extraction happens here so the batch itself
        does not have to be held until the end of the run.

        :returns: ``{"preds", "obs", "obsm"}``, consumed by
            :func:`~sckitflow._predict.predictions_to_adata`.
        """
        if self.inference_method is None:
            raise ValueError("this plan has no inference method: pass one to `TrainingPlan(...)`.")
        step_data, leaf = batch
        generator, _ = generators(self.seed, _PREDICT, dataloader_idx, batch_idx, device=self.device)
        preds = self.inference_method.predict(step_data, generator=generator, **self._predict_kwargs)
        return prediction_record(self.trainer.predict_dataloaders, step_data, leaf, preds)

    def configure_optimizers(self) -> Any:
        if self._optimizer is None:
            raise ValueError("this plan was built without an optimizer, so it can only predict.")
        if self._lr_scheduler is None:
            return self._optimizer
        return {
            "optimizer": self._optimizer,
            "lr_scheduler": {"scheduler": self._lr_scheduler, "interval": self._lr_scheduler_interval},
        }
