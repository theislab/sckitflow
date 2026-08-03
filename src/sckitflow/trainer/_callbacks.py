from collections.abc import Callable, Sequence
from copy import deepcopy
from typing import Any

import lightning.pytorch as pl
import numpy as np
import torch

__all__ = ["MetricsCallback"]


class MetricsCallback(pl.Callback):
    """Computes validation metrics from the predictions and targets of every node.

    Metrics are accumulated across the nodes of a validation set and computed once at the
    end of the run, then logged as ``"{val_id}_{metric_name}"`` -- which is what
    :class:`DataFrameLogger` keys on to route them into the right frame, and what any
    other Lightning logger will show.

    Each validation dataloader gets its own copy of every metric, so two validation sets
    never pool their state.
    """

    def __init__(
        self,
        metrics: dict[str, Any],
        transforms: Callable | Sequence[Callable] | None = None,
        pred_transforms: Callable | Sequence[Callable] | None = None,
        target_transforms: Callable | Sequence[Callable] | None = None,
    ) -> None:
        """Initializes the metrics callback.

        :param metrics: Mapping of metric name to an instantiated metric object. Each
            metric must implement ``update(pred, target)`` and ``compute()``; a
            :class:`torchmetrics.Metric` satisfies this.
        :type metrics: class: `dict[str, Any]`

        :param transforms: (Optional) Transform(s) applied to both predictions and
            targets. Defaults to `None`.
        :type transforms: class: `Callable | Sequence[Callable] | None`

        :param pred_transforms: (Optional) Transform(s) applied only to predictions,
            overriding :param: `transforms`. Defaults to `None`.
        :type pred_transforms: class: `Callable | Sequence[Callable] | None`

        :param target_transforms: (Optional) Transform(s) applied only to targets,
            overriding :param: `transforms`. Defaults to `None`.
        :type target_transforms: class: `Callable | Sequence[Callable] | None`
        """
        super().__init__()
        self._metrics = metrics
        self._pred_transforms = self._to_list(pred_transforms if pred_transforms is not None else transforms)
        self._target_transforms = self._to_list(target_transforms if target_transforms is not None else transforms)

        # Per-dataloader metric copies, built lazily on first use.
        self._per_loader: dict[int, dict[str, Any]] = {}

    @staticmethod
    def _to_list(x: Any) -> list[Callable]:
        if x is None:
            return []
        if callable(x):
            return [x]
        return list(x)

    @staticmethod
    def _apply_transforms(data: Any, transforms: list[Callable]) -> Any:
        for t in transforms:
            data = t(data)
        return data

    @staticmethod
    def _to_tensor(data: Any, device: torch.device) -> torch.Tensor:
        """Converts array-likes to a tensor on ``device``."""
        if isinstance(data, torch.Tensor):
            return data.to(device)
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data).float().to(device)
        if hasattr(data, "__array__"):
            return torch.from_numpy(np.asarray(data)).float().to(device)
        return torch.tensor(data, dtype=torch.float32, device=device)

    def _metrics_for(self, dataloader_idx: int, device: torch.device) -> dict[str, Any]:
        """Returns this dataloader's metric copies, creating them on first access.

        Every copy is cloned from :attr: `_metrics`, which is never updated and so stays
        pristine. Cloning one dataloader's metrics from another's would carry over
        whatever that dataloader had already accumulated.
        """
        if dataloader_idx not in self._per_loader:
            self._per_loader[dataloader_idx] = {
                name: clone.to(device) if hasattr(clone, "to") else clone
                for name, clone in deepcopy(self._metrics).items()
            }
        return self._per_loader[dataloader_idx]

    def _val_id(self, trainer: pl.Trainer, dataloader_idx: int) -> str:
        """Resolves the name of a validation set from its dataloader position."""
        val_ids = getattr(trainer, "val_ids", None) or []
        if dataloader_idx < len(val_ids):
            return val_ids[dataloader_idx]
        return f"val{dataloader_idx}"

    def on_validation_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Clears accumulated state so each validation run reports only its own nodes."""
        for metrics in self._per_loader.values():
            for metric in metrics.values():
                if hasattr(metric, "reset"):
                    metric.reset()

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: dict[str, Any] | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Feeds one node's predictions and targets into this dataloader's metrics."""
        if not outputs:
            return
        preds = outputs.get("predictions")
        targets = outputs.get("targets")
        if preds is None or targets is None:
            return

        preds = self._apply_transforms(preds, self._pred_transforms)
        targets = self._apply_transforms(targets, self._target_transforms)

        device = pl_module.device
        preds = self._to_tensor(preds, device)
        targets = self._to_tensor(targets, device)

        for metric in self._metrics_for(dataloader_idx, device).values():
            metric.update(preds, targets)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Computes every metric and logs it as ``"{val_id}_{name}"``."""
        for dataloader_idx, metrics in self._per_loader.items():
            val_id = self._val_id(trainer, dataloader_idx)
            for name, metric in metrics.items():
                value = metric.compute()
                if hasattr(value, "item"):
                    value = value.item()
                # The value is already reduced over every node, so the reduction weight is
                # irrelevant -- `batch_size=1` pins it and keeps Lightning from trying to
                # infer a size from a node, which it cannot walk.
                pl_module.log(f"{val_id}_{name}", value, on_step=False, on_epoch=True, batch_size=1)

    @property
    def metrics(self) -> dict[str, Any]:
        """Exposes the :param metrics: attribute set at initialization."""
        return self._metrics
