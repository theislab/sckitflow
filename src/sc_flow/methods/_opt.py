import abc
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

__all__ = ["BaseOptManager", "OptimConfig"]


class BaseOptManager(abc.ABC):
    @abc.abstractmethod
    def step(self, step_fn: Callable[[Any, ...], Any], node: Any, *args, **kwargs) -> Any:
        """Perform one optimization step."""
        pass


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
