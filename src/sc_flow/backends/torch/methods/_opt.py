import torch

from sc_flow.backends.torch.nn._modules import BaseModule
from sc_flow.methods._opt import BaseOptManager, OptimConfig

__all__ = ["TorchOptimizationManager"]


class TorchOptimizationManager(BaseOptManager):
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
    def from_config(cls, module: BaseModule, config: OptimConfig) -> "TorchOptimizationManager":
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
    def lr_scheduler(self) -> torch.optim.Optimizer:
        return self._lr_scheduler

    @property
    def lr_scheduler_step(self) -> torch.optim.Optimizer:
        return self._lr_scheduler_step
