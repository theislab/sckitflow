from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch

__all__ = ["OptimConfig"]

# Lightning expects `"step"` or `"epoch"`; `"train_step"` is accepted as the historical
# spelling of `"step"`.
_INTERVAL_ALIASES = {"train_step": "step", "step": "step", "epoch": "epoch"}


@dataclass
class OptimConfig:
    """Declarative optimizer/scheduler configuration.

    Resolved into the mapping :meth:`~lightning.pytorch.LightningModule.configure_optimizers`
    expects, so Lightning owns ``zero_grad``/``backward``/``step`` and the scheduler
    interval.
    """

    # Either provide an already instantiated optimizer, OR provide class/string + kwargs
    optimizer: Any | None = None  # pre-created optimizer instance
    optimizer_cls: str | type | None = None  # class or name resolved against `torch.optim`
    optimizer_kwargs: dict[str, Any] | None = None
    lr: float = 5e-5

    # Similarly for scheduler
    lr_scheduler: Any | None = None  # pre-created scheduler instance
    lr_scheduler_cls: str | type | None = None
    lr_scheduler_kwargs: dict[str, Any] | None = None
    lr_scheduler_step: str = "train_step"
    lr_scheduler_monitor: str | None = None

    def __post_init__(self):
        if self.optimizer_kwargs is None:
            self.optimizer_kwargs = {}
        if self.lr_scheduler_kwargs is None:
            self.lr_scheduler_kwargs = {}
        if self.lr_scheduler_step not in _INTERVAL_ALIASES:
            raise ValueError(
                f"lr_scheduler_step must be one of {sorted(_INTERVAL_ALIASES)}, got {self.lr_scheduler_step!r}"
            )

    @property
    def lr_scheduler_interval(self) -> str:
        """The scheduler interval in Lightning's spelling (``"step"`` or ``"epoch"``)."""
        return _INTERVAL_ALIASES[self.lr_scheduler_step]

    @staticmethod
    def _resolve_cls(cls_or_name: str | type, namespace: Any, kind: str) -> type:
        """Resolves a class either directly or by name against a ``torch.optim`` namespace."""
        if isinstance(cls_or_name, str):
            if not hasattr(namespace, cls_or_name):
                raise ValueError(f"{kind} '{cls_or_name}' not found in {namespace.__name__}")
            return getattr(namespace, cls_or_name)
        if not callable(cls_or_name):
            raise TypeError(f"{kind} must be a string or callable, got {type(cls_or_name)}")
        return cls_or_name

    def build_optimizer(self, params: Iterable[torch.nn.Parameter]) -> torch.optim.Optimizer:
        """Returns the configured optimizer, building it over ``params`` when not pre-created."""
        if self.optimizer is not None:
            return self.optimizer
        opt_cls = self._resolve_cls(self.optimizer_cls or "Adam", torch.optim, "Optimizer")
        return opt_cls(params, lr=self.lr, **self.optimizer_kwargs)

    def build_lr_scheduler(self, optimizer: torch.optim.Optimizer) -> Any | None:
        """Returns the configured scheduler, or `None` when none was requested."""
        if self.lr_scheduler is not None:
            return self.lr_scheduler
        if self.lr_scheduler_cls is None:
            return None
        sched_cls = self._resolve_cls(self.lr_scheduler_cls, torch.optim.lr_scheduler, "Scheduler")
        return sched_cls(optimizer, **self.lr_scheduler_kwargs)

    def resolve(self, params: Iterable[torch.nn.Parameter]) -> dict[str, Any]:
        """Builds the :meth:`configure_optimizers` return value for these settings."""
        optimizer = self.build_optimizer(params)
        scheduler = self.build_lr_scheduler(optimizer)
        if scheduler is None:
            return {"optimizer": optimizer}

        lr_scheduler_config: dict[str, Any] = {
            "scheduler": scheduler,
            "interval": self.lr_scheduler_interval,
        }
        # Only schedulers that key off a metric (e.g. `ReduceLROnPlateau`) need a monitor,
        # and Lightning rejects a `None` monitor.
        if self.lr_scheduler_monitor is not None:
            lr_scheduler_config["monitor"] = self.lr_scheduler_monitor
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
