import abc
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any

__all__ = ["OptimConfig", "BaseOptManager"]


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

    # automatic mixed precision
    use_amp: bool = False
    scaler_kwargs: dict[str, Any] = dc_field(default_factory=lambda: {})

    # gradient accumulation
    n_grad_acc_steps: int = 1

    # ema
    use_ema: bool = False
    ema_decay: float = 0.9999

    def __post_init__(self):
        if self.optimizer_kwargs is None:
            self.optimizer_kwargs = {}
        if self.lr_scheduler_kwargs is None:
            self.lr_scheduler_kwargs = {}
        if self.scaler_kwargs is None:
            self.scaler_kwargs = {}


class BaseOptManager(abc.ABC):
    @abc.abstractmethod
    def step(self, step_fn: Callable[[Any, ...], Any], node: Any, *args, **kwargs) -> dict[str, Any]:
        """Perform one optimization step."""
        pass

    @classmethod
    @abc.abstractmethod
    def from_config(cls, module: Any, config: OptimConfig) -> "BaseOptManager":
        """Initializes the optimization manager from the specified configurations."""
        pass
