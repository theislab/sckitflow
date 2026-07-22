"""Optional PyTorch Lightning backend for the training loop.

This module is imported lazily (only when ``trainer.framework == "lightning"``)
so ``lightning`` stays an optional dependency. It adapts the existing
:class:`~sc_flow.methods._methods.BaseMethod` and samplers onto Lightning:

* :class:`LitSCFlowModule` wraps a ``BaseMethod`` as a ``LightningModule`` using
  **manual optimization**, preserving the native loop's per-node optimizer step.
* :class:`SCFlowDataModule` exposes the step-based samplers as an
  ``IterableDataset`` so Lightning can drive them with ``max_steps``.
* :func:`build_lightning_trainer` maps a :class:`TrainerConfig` onto ``pl.Trainer``.

The method keeps ownership of device placement (it moves batch tensors to its own
``device_id``), so batch transfer is a no-op and the Lightning ``accelerator`` is
derived from ``trainer.device``.
"""

from __future__ import annotations

import warnings
from typing import Any

import lightning.pytorch as pl
from torch.utils.data import DataLoader, IterableDataset

from sc_flow.config._run import TrainerConfig
from sc_flow.methods._methods import BaseMethod
from sc_flow.methods._opt import OptimConfig

__all__ = ["LitSCFlowModule", "SCFlowDataModule", "build_lightning_trainer", "fit_with_lightning"]


class _SamplerIterableDataset(IterableDataset):
    """Yields ``sampler.sample()`` (a batch of nodes) up to ``n_steps`` times."""

    def __init__(self, sampler: Any, n_steps: int) -> None:
        super().__init__()
        self._sampler = sampler
        self._n_steps = n_steps

    def __iter__(self):
        for _ in range(self._n_steps):
            yield self._sampler.sample()


class SCFlowDataModule(pl.LightningDataModule):
    """Adapts the step-based samplers to Lightning dataloaders.

    ``batch_size=None`` disables collation: each item produced by the sampler
    (already a batch of nodes) is passed through to ``training_step`` unchanged.
    """

    def __init__(self, train_sampler: Any, n_steps: int, val_samplers_dict: dict[str, Any] | None = None) -> None:
        super().__init__()
        self._train_sampler = train_sampler
        self._n_steps = n_steps
        self._val_samplers = val_samplers_dict or {}

    def train_dataloader(self) -> DataLoader:
        ds = _SamplerIterableDataset(self._train_sampler, self._n_steps)
        return DataLoader(ds, batch_size=None, collate_fn=lambda x: x)


class LitSCFlowModule(pl.LightningModule):
    """Wrap a :class:`BaseMethod` as a ``LightningModule`` (manual optimization)."""

    def __init__(self, method: BaseMethod, optim_config: OptimConfig, grad_clip: float | None = None) -> None:
        super().__init__()
        self.automatic_optimization = False
        self._method = method
        self._optim_config = optim_config
        self._grad_clip = grad_clip
        # Register the underlying network so Lightning tracks its parameters for
        # checkpointing/optimization. Device placement stays with the method.
        self._net = method.module

    def configure_optimizers(self):
        from sc_flow.backends.torch.methods._opt import TorchOptimizationManager

        mgr = TorchOptimizationManager.from_config(self._method.module, self._optim_config)
        if mgr.lr_scheduler is not None:
            return {"optimizer": mgr.optimizer, "lr_scheduler": mgr.lr_scheduler}
        return mgr.optimizer

    def transfer_batch_to_device(self, batch: Any, device: Any, dataloader_idx: int) -> Any:
        # The method moves the tensors it needs onto its own device; leave the
        # custom node objects untouched.
        return batch

    def training_step(self, batch: Any, batch_idx: int) -> None:
        opt = self.optimizers()
        last_logs: dict[str, Any] = {}
        for node in batch:
            loss, logs = self._method.train_step(node)
            opt.zero_grad()
            self.manual_backward(loss)
            if self._grad_clip is not None:
                self.clip_gradients(opt, gradient_clip_val=self._grad_clip)
            opt.step()
            last_logs = logs
        sched = self.lr_schedulers()
        if sched is not None:
            sched.step()
        for key, value in last_logs.items():
            if isinstance(value, int | float):
                self.log(key, float(value), prog_bar=True, batch_size=1)
        return None


def _accelerator_devices(device: str) -> tuple[str, Any]:
    """Map a ``trainer.device`` string to Lightning ``(accelerator, devices)``."""
    key = device.lower()
    if key == "cpu":
        return "cpu", 1
    if key in ("cuda", "gpu"):
        return "gpu", 1
    if key == "mps":
        return "mps", 1
    # Fall back to letting Lightning auto-detect.
    return device, "auto"


def build_lightning_trainer(cfg: TrainerConfig) -> pl.Trainer:
    """Construct a ``pl.Trainer`` from a :class:`TrainerConfig`."""
    accelerator, devices = _accelerator_devices(cfg.device)
    kwargs: dict[str, Any] = {
        "max_steps": cfg.n_train_steps,
        "accelerator": accelerator,
        "devices": devices,
        "enable_checkpointing": False,
        "logger": False,
    }
    if cfg.precision is not None:
        kwargs["precision"] = cfg.precision
    if cfg.log_every_n_steps is not None:
        kwargs["log_every_n_steps"] = cfg.log_every_n_steps
    if cfg.accumulate_grad_batches != 1:
        # Manual optimization ignores Trainer-level gradient accumulation.
        warnings.warn(
            "accumulate_grad_batches is ignored by the Lightning backend "
            "(manual optimization); accumulate inside the method instead.",
            RuntimeWarning,
            stacklevel=2,
        )
    kwargs.update(cfg.trainer_kwargs)
    return pl.Trainer(**kwargs)


def fit_with_lightning(
    method: BaseMethod,
    optim_config: OptimConfig,
    trainer_cfg: TrainerConfig,
    train_sampler: Any,
    val_samplers_dict: dict[str, Any] | None = None,
) -> pl.Trainer:
    """Run training through a ``pl.Trainer`` and return it.

    The LightningModule and DataModule come from the method's harness seam
    (:meth:`~sc_flow.methods._methods.BaseMethod.make_lightning_module` /
    ``make_datamodule``), so a JAX-compute method returns a ``CellFlowJaxModule``
    here exactly where a torch method returns a ``LitSCFlowModule`` — both train
    through this one entry point.
    """
    lit = method.make_lightning_module(optim_config, grad_clip=trainer_cfg.gradient_clip_val)
    datamodule = method.make_datamodule(train_sampler, trainer_cfg.n_train_steps, val_samplers_dict)
    trainer = build_lightning_trainer(trainer_cfg)
    trainer.fit(lit, datamodule=datamodule)
    return trainer
