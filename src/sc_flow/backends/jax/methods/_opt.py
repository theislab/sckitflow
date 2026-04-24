from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import optax
from flax.training import train_state
from optax import GradientTransformation, Schedule

from sc_flow.methods._opt import BaseOptManager, OptimConfig

__all__ = ["JaxOptimizationManager", "TrainStateWithBatchStats"]

PyTree = Any


class TrainStateWithBatchStats(train_state.TrainState):
    batch_stats: Any


class JaxOptimizationManager(BaseOptManager):
    def __init__(
        self,
        tx: GradientTransformation,
        lr_scheduler_step: str = "train_step",
    ) -> None:
        self._tx = tx
        self._lr_scheduler_step = lr_scheduler_step

    def step(
        self,
        optim_data: tuple[train_state.TrainState, Callable[[PyTree], tuple[Any, dict]]],
    ) -> tuple[train_state.TrainState, dict]:
        ts, loss_fn = optim_data
        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(ts.params)
        new_batch_stats = metrics.pop("batch_stats", None)
        ts = ts.apply_gradients(grads=grads)
        if new_batch_stats is not None:
            ts = ts.replace(batch_stats=new_batch_stats)
        return ts, metrics

    def create_train_state(
        self,
        apply_fn: Callable | None,
        params: PyTree,
        batch_stats: PyTree | None = None,
    ) -> train_state.TrainState:
        if batch_stats is not None:
            return TrainStateWithBatchStats.create(
                apply_fn=apply_fn,
                params=params,
                tx=self._tx,
                batch_stats=batch_stats,
            )
        return train_state.TrainState.create(
            apply_fn=apply_fn,
            params=params,
            tx=self._tx,
        )

    @classmethod
    def from_config(cls, config: OptimConfig) -> JaxOptimizationManager:
        if config.optimizer is not None:
            tx = config.optimizer
        else:
            learning_rate = cls._resolve_schedule(config)
            opt_cls = cls._resolve_optimizer_cls(config.optimizer_cls)
            tx = opt_cls(learning_rate=learning_rate, **config.optimizer_kwargs)

        return cls(tx, lr_scheduler_step=config.lr_scheduler_step)

    @staticmethod
    def _resolve_schedule(config: OptimConfig) -> float | Schedule:
        if config.lr_scheduler is not None:
            return config.lr_scheduler

        if config.lr_scheduler_cls is not None:
            sched_cls = config.lr_scheduler_cls
            if isinstance(sched_cls, str):
                if not hasattr(optax, sched_cls):
                    raise ValueError(
                        f"Schedule '{sched_cls}' not found in optax. "
                        "Pass the callable directly or check the optax docs."
                    )
                sched_cls = getattr(optax, sched_cls)
            elif not callable(sched_cls):
                raise TypeError(f"lr_scheduler_cls must be a string or callable, got {type(sched_cls)}")
            return sched_cls(**config.lr_scheduler_kwargs)

        return config.lr

    @staticmethod
    def _resolve_optimizer_cls(optimizer_cls: str | type | None) -> type:
        if optimizer_cls is None:
            return optax.adam

        if isinstance(optimizer_cls, str):
            name = optimizer_cls.lower()
            if not hasattr(optax, name):
                raise ValueError(
                    f"Optimizer '{optimizer_cls}' not found in optax. "
                    "Pass the callable directly or check the optax docs."
                )
            return getattr(optax, name)

        if callable(optimizer_cls):
            return optimizer_cls

        raise TypeError(f"optimizer_cls must be a string or callable, got {type(optimizer_cls)}")

    @property
    def tx(self) -> GradientTransformation:
        return self._tx

    @property
    def lr_scheduler_step(self) -> str:
        return self._lr_scheduler_step
