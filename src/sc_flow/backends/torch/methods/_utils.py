from dataclasses import dataclass
from typing import Any, Literal

import torch

from sc_flow.backends.torch.nn._modules import BaseModule

__all__ = ["StepData", "OptimizationManager", "PredictionData"]


@dataclass
class StepData:
    target_state: torch.Tensor
    target_coupling_lin: torch.Tensor
    target_coupling_quad: torch.Tensor | None
    target_condition_data: Any | None
    target_group_data: Any | None
    source_state: torch.Tensor | None
    source_coupling_lin: torch.Tensor | None
    source_coupling_quad: torch.Tensor | None
    source_condition_data: Any | None
    source_group_data: Any | None


@dataclass
class OptimizationManager:
    optimizer: torch.optim.Optimizer
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
    lr_scheduler_step: Literal["train_step", "epoch"] = "train_step"
    plan_kwargs: dict[str, Any] | None = None

    def backward_pass(self, loss: torch.Tensor) -> None:
        # optimizer step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # optional scheduler step
        if self.lr_scheduler is not None and self.lr_scheduler_step == "train_step":
            self.lr_scheduler.step()

    @classmethod
    def init_from_module(
        cls,
        module: BaseModule,
        optimizer_cls: type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: dict[str, Any] | None = None,
        lr: float = 5e-5,
        lr_scheduler_cls: type[torch.optim.lr_scheduler.LRScheduler] | None = None,
        lr_scheduler_kwargs: dict[str, Any] | None = None,
        lr_scheduler_step: Literal["train_step", "validation_step"] = "train_step",
        plan_kwargs: dict[str, Any] | None = None,
    ) -> "OptimizationManager":
        """"""  # noqa

        # initialize optimizer
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
        optimizer = optimizer_cls(
            module.parameters(),
            lr=lr,
            **optimizer_kwargs,
        )

        # optional initialization of lr scheduler
        if lr_scheduler_cls is not None:
            if lr_scheduler_kwargs is None:
                lr_scheduler_kwargs = {}
            lr_scheduler = lr_scheduler_cls(optimizer, **lr_scheduler_kwargs)
        else:
            lr_scheduler = None

        return cls(
            optimizer,
            lr_scheduler=lr_scheduler,
            lr_scheduler_step=lr_scheduler_step,
            plan_kwargs=plan_kwargs if plan_kwargs is not None else {},
        )


@dataclass
class PredictionData:
    samples: torch.Tensor
    traj: torch.Tensor | None = None
