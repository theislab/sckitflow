from dataclasses import dataclass
from typing import Any

import torch

from sckitflow.core.nn._modules import BaseModule

__all__ = ["OptimConfig", "OptimizationManager"]


@dataclass
class OptimConfig:
    """Configuration for creating an optimization manager."""

    # Either provide an already instantiated optimizer, OR provide class/string + kwargs
    optimizer: Any | None = None  # pre‑created optimizer instance
    optimizer_cls: str | type | None = None  # class or string name (resolved by backend)
    optimizer_kwargs: dict[str, Any] | None = None
    lr: float = 5e-5

    # Similarly for scheduler
    lr_scheduler: Any | None = None  # pre‑created scheduler instance
    lr_scheduler_cls: str | type | None = None
    lr_scheduler_kwargs: dict[str, Any] | None = None
    lr_scheduler_step: str = "train_step"

    plan_kwargs: dict[str, Any] | None = None

    def __post_init__(self):
        if self.optimizer_kwargs is None:
            self.optimizer_kwargs = {}
        if self.lr_scheduler_kwargs is None:
            self.lr_scheduler_kwargs = {}
        if self.plan_kwargs is None:
            self.plan_kwargs = {}


class OptimizationManager:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        lr_scheduler_step: str = "train_step",
    ) -> None:
        self._optimizer = optimizer
        self._lr_scheduler = lr_scheduler
        self._lr_scheduler_step = lr_scheduler_step

    def step(self, loss: torch.Tensor) -> None:
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        if self._lr_scheduler is not None and self._lr_scheduler_step == "train_step":
            self._lr_scheduler.step()

    @classmethod
    def from_config(cls, module: BaseModule, config: OptimConfig) -> "OptimizationManager":
        # Resolve optimizer
        if config.optimizer is not None:
            optimizer = config.optimizer
        else:
            opt_cls = config.optimizer_cls
            if opt_cls is None:
                opt_cls = "Adam"
            if isinstance(opt_cls, str):
                # Resolve string to torch.optim class
                if not hasattr(torch.optim, opt_cls):
                    raise ValueError(f"Optimizer '{opt_cls}' not found in torch.optim")
                opt_cls = getattr(torch.optim, opt_cls)
            elif not callable(opt_cls):
                raise TypeError(f"optimizer_cls must be a string or callable, got {type(opt_cls)}")
            optimizer = opt_cls(module.parameters(), lr=config.lr, **config.optimizer_kwargs)

        # Resolve scheduler
        scheduler = config.lr_scheduler
        if scheduler is None and config.lr_scheduler_cls is not None:
            sched_cls = config.lr_scheduler_cls
            if isinstance(sched_cls, str):
                if not hasattr(torch.optim.lr_scheduler, sched_cls):
                    raise ValueError(f"Scheduler '{sched_cls}' not found in torch.optim.lr_scheduler")
                sched_cls = getattr(torch.optim.lr_scheduler, sched_cls)
            elif not callable(sched_cls):
                raise TypeError(f"lr_scheduler_cls must be a string or callable, got {type(sched_cls)}")
            scheduler = sched_cls(optimizer, **config.lr_scheduler_kwargs)

        return cls(optimizer, lr_scheduler=scheduler, lr_scheduler_step=config.lr_scheduler_step)

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self._optimizer

    @property
    def lr_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler | None:
        return self._lr_scheduler

    @property
    def lr_scheduler_step(self) -> str:
        return self._lr_scheduler_step
