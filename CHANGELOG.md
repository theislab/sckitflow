# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [Unreleased]

### Added

- Basic tool, preprocessing and plotting functions
- Training runs on PyTorch Lightning. Methods are `LightningModule`s, `Trainer` is a thin
  `lightning.pytorch.Trainer` subclass, and checkpointing, early stopping, AMP, gradient
  clipping/accumulation and multi-device training come with it.
- `DataFrameLogger`, an in-memory `lightning.pytorch.loggers.Logger` backing
  `Trainer.get_train_logs_df()` / `get_val_logs_df()`.
- `Model.train` accepts `val_predict_kwargs`, and forwards any extra keyword arguments to
  the trainer (`accelerator`, `precision`, `gradient_clip_val`, ...).

### Changed

- Training steps are counted in *node-steps*: the train sampler yields one node at a time
  and each node is one batch and one optimizer step. The gradient schedule is unchanged --
  it was already one step per node -- but a run that used `n_train_steps=N` with
  `n_nodes=k` now needs `n_train_steps=N * k`.
- `TrainSampler.__iter__` yields individual nodes instead of tuples, and is re-iterable
  and unbounded by default; `max_iter_steps` defaults to `None` so the trainer's
  `n_train_steps` governs run length.
- The torch-only layer is flat: `BaseMethod`/`GenerativeFlow` and `register_method`
  moved from `sckitflow.methods` to `sckitflow.core.methods`, merging the former
  backend-agnostic layer with `TorchBaseMethod`/`TorchGenerativeFlow`.
- The method subclass contract is `compute_loss` (the loss) and `infer` (inference),
  renamed from `_step_fn` and `_predict`.
- `OptimConfig` moved to `sckitflow.core.methods` and resolves itself into the mapping
  `configure_optimizers` returns; it gained `lr_scheduler_monitor`.
- Methods read `self.device`/`self.dtype` from Lightning rather than storing
  `device_id`/`dtype`, so device placement follows the trainer. `device_id` now defaults
  to `None` (leave in place) instead of eagerly selecting CUDA.
- `Model.save` no longer writes the trainer into the pickle, so training logs and
  optimizer state do not survive a round trip; use
  `lightning.pytorch.callbacks.ModelCheckpoint` for those.

### Removed

- The backend-agnostic indirection: `sckitflow.methods`, `BaseOptManager`,
  `TorchOptimizationManager`, `BaseMethod.set_train_mode`, the `backend` argument to
  `register_method`, and the jax branches in `_runtime.py`.
- The custom callback hierarchy -- `BaseCallback`, `ComputationalCallback`,
  `LoggingCallback`, `TrainingCallbacks` and `WandBLogger`. Callbacks are now
  `lightning.pytorch.Callback`s, `MetricsCallback` included, and W&B logging goes through
  `lightning.pytorch.loggers.WandbLogger`.
