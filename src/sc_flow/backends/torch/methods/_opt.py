from collections.abc import Callable
from typing import Any

import torch

from sc_flow.backends.torch.methods._ema import ExponentialMovingAverage
from sc_flow.backends.torch.nn._modules import BaseModule
from sc_flow.methods._opt import BaseOptManager, OptimConfig

__all__ = ["TorchOptimizationManager"]


class TorchOptimizationManager(BaseOptManager):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        lr_scheduler_step: str = "train_step",
        use_amp: bool = False,
        scaler_kwargs: dict[str, Any] | None = None,
        n_grad_acc_steps: int = 1,
        ema: ExponentialMovingAverage | None = None,
    ) -> None:
        self._optimizer = optimizer
        self._lr_scheduler = lr_scheduler
        self._lr_scheduler_step = lr_scheduler_step
        self._use_amp = use_amp
        self._n_grad_acc_steps = n_grad_acc_steps
        self._ema = ema

        # prepare scaler when using amp
        if self._use_amp:
            scaler_kwargs = {} if scaler_kwargs is None else scaler_kwargs
            device = scaler_kwargs.pop("device", self._device)
            self._scaler = torch.amp.GradScaler(device=device, **scaler_kwargs)
        else:
            self._scaler = None

    @property
    def _device(self) -> torch.device:
        """Get device from the optimizer's parameters."""
        if self._optimizer.param_groups:
            for param in self._optimizer.param_groups[0]["params"]:
                if hasattr(param, "device") and param.device is not None:
                    return param.device.type
        return "cpu"

    def _call_step_fn(self, step_fn: Callable[[Any, ...], Any], node: Any, *args, **kwargs) -> Any:
        if self._use_amp:
            with torch.amp.autocast(self._device):
                return step_fn(node, *args, **kwargs)
        else:
            return step_fn(node, *args, **kwargs)

    def step(self, step_idx: int, step_fn: Callable[[Any, ...], Any], node: Any, *args, **kwargs) -> Any:
        # call step fn
        loss, step_dict = self._call_step_fn(
            step_fn,
            node,
            *args,
            **kwargs,
        )

        # scale loss by gradient acc steps
        loss = loss / self._n_grad_acc_steps

        # optionally scaling the loss
        if self._scaler is not None:
            self._scaler.scale(loss).backward()
        else:
            loss.backward()

        # optimizer step when grad accumulation done
        if step_idx % self._n_grad_acc_steps == 0:
            # optionally wrap optimizer step with scaler
            if self._scaler is not None:
                self._scaler.step(self._optimizer)
                self._scaler.update()
            else:
                self._optimizer.step()

            # zeroing gradients
            self._optimizer.zero_grad()

            # optional ema step
            if self._ema is not None:
                self._ema.update()

        # lr scheduler step
        if self._lr_scheduler is not None and self._lr_scheduler_step == "train_step":
            self._lr_scheduler.step()
        return step_dict

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

        # initialize ema
        if config.use_ema:
            ema = ExponentialMovingAverage(module, decay=config.ema_decay)
        else:
            ema = None

        return cls(
            optimizer,
            lr_scheduler=scheduler,
            lr_scheduler_step=config.lr_scheduler_step,
            use_amp=config.use_amp,
            scaler_kwargs=config.scaler_kwargs,
            n_grad_acc_steps=config.n_grad_acc_steps,
            ema=ema,
        )

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self._optimizer

    @property
    def lr_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler | None:
        return self._lr_scheduler

    @property
    def lr_scheduler_step(self) -> str:
        return self._lr_scheduler_step
