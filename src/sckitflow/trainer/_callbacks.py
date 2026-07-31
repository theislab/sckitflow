from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from sckitflow.trainer._trainer import Trainer
from sckitflow._types import PredictionData

__all__ = [
    "BaseCallback",
    "ComputationalCallback",
    "LoggingCallback",
    "TrainingCallbacks",
    "MetricsCallback",
    "WandBLogger",
]


class BaseCallback:
    """Base hooks shared by every callback.

    Do not subclass this directly -- pick :class:`ComputationalCallback` or
    :class:`LoggingCallback`, which is how :class:`TrainingCallbacks` decides
    what each callback is sent.
    """

    def on_train_begin(self, trainer: "Trainer", **kwargs) -> None:
        pass

    def on_train_step(self, trainer: "Trainer", step: int, logs: dict[str, Any], **kwargs) -> None:
        pass

    def on_valid_step(
        self, trainer: "Trainer", step: int, val_id: str, predictions_dict: dict[str, Any], **kwargs
    ) -> dict[str, Any] | None:
        return None

    def on_train_end(self, trainer: "Trainer", **kwargs) -> None:
        pass


class ComputationalCallback(BaseCallback):
    """Derives values from the raw validation predictions.

    Receives ``predictions_dict`` on `on_valid_step` and returns metrics.
    """


class LoggingCallback(BaseCallback):
    """Reports already-computed values; never produces metrics itself.

    Receives the metrics the computational callbacks returned, and is the only
    kind sent `on_train_step`.
    """


class MetricsCallback(ComputationalCallback):
    """Computes metrics from predictions and targets using pre-instantiated metrics."""

    def __init__(
        self,
        metrics: dict[str, Any],
        backend: Literal["torch", "jax"] = "torch",
        transforms: Callable | Sequence[Callable] | None = None,
        pred_transforms: Callable | Sequence[Callable] | None = None,
        target_transforms: Callable | Sequence[Callable] | None = None,
        device: str = "cpu",
    ) -> None:
        """Initializes the metrics callback.

        Args:
            metrics: Dictionary mapping metric names to instantiated metric objects.
                    Each metric must have `update(pred, target)` and `compute()` methods.
            backend: Backend to use ("torch" or "jax").
            transforms: Optional transform(s) applied to both predictions and targets.
            pred_transforms: Optional transform(s) applied only to predictions.
            target_transforms: Optional transform(s) applied only to targets.
            device: Device for tensor operations (PyTorch only).
        """
        self._metrics = metrics
        self._backend = backend
        self._device = device
        self._pred_transforms = self._to_list(pred_transforms if pred_transforms is not None else transforms)
        self._target_transforms = self._to_list(target_transforms if target_transforms is not None else transforms)

        # Lazy imports for backend modules
        self._torch = None
        self._jax = None

    def _get_torch(self):
        """Lazy import torch."""
        if self._torch is None:
            import torch

            self._torch = torch
        return self._torch

    def _reset_metrics(self) -> None:
        """Reset all metrics."""
        for metric in self._metrics.values():
            if hasattr(metric, "reset"):
                metric.reset()

    @staticmethod
    def _to_list(x: Any) -> list[Callable]:
        if x is None:
            return []
        if callable(x):
            return [x]
        return list(x)

    def _apply_transforms(self, data: Any, transforms: list[Callable]) -> Any:
        for t in transforms:
            data = t(data)
        return data

    def _to_tensor(self, data: Any) -> Any:
        """Convert various array types to backend tensor."""
        if isinstance(data, PredictionData):
            data = data.X
        if self._backend == "torch":
            torch = self._get_torch()
            if isinstance(data, torch.Tensor):
                return data.to(self._device)
            if isinstance(data, np.ndarray):
                return torch.from_numpy(data).float().to(self._device)
            if hasattr(data, "__array__"):
                return torch.from_numpy(np.array(data)).float().to(self._device)
            return torch.tensor(data, dtype=torch.float32).to(self._device)
        elif self._backend == "jax":
            # For JAX, we keep as numpy arrays
            if isinstance(data, np.ndarray):
                return data
            if hasattr(data, "__array__"):
                return np.array(data)
            return np.asarray(data)
        else:
            raise ValueError(f"Unsupported backend: {self._backend}")

    def on_valid_step(
        self, trainer: "Trainer", step: int, val_id: str, predictions_dict: dict[str, Any], **kwargs
    ) -> dict[str, float]:
        # Reset all metrics
        self._reset_metrics()

        # Process each perturbation
        for _pert, data in predictions_dict.items():
            preds = data.get("predictions")
            targets = data.get("targets")
            if preds is None or targets is None:
                continue

            # Apply transforms
            preds = self._apply_transforms(preds, self._pred_transforms)
            targets = self._apply_transforms(targets, self._target_transforms)

            # Convert to backend tensors
            preds = self._to_tensor(preds)
            targets = self._to_tensor(targets)

            # Update all metrics
            for metric in self._metrics.values():
                metric.update(preds, targets)

        # Compute final metrics
        all_metrics = {}
        for name, metric in self._metrics.items():
            value = metric.compute()
            if hasattr(value, "item"):
                value = value.item()
            all_metrics[f"{val_id}_{name}"] = value

        return all_metrics


class WandBLogger(LoggingCallback):
    """Logs metrics to Weights & Biases."""

    def __init__(
        self,
        project_name: str,
        log_dir: str,
        config: dict[str, Any],
        **kwargs,
    ) -> None:
        self.project_name = project_name
        self.log_dir = log_dir
        self.config = config
        self.kwargs = kwargs
        self._wandb = None
        self._omegaconf = None

    def _get_wandb(self):
        if self._wandb is None:
            import wandb

            self._wandb = wandb
        return self._wandb

    def _get_omegaconf(self):
        if self._omegaconf is None:
            import omegaconf

            self._omegaconf = omegaconf
        return self._omegaconf

    def on_train_begin(self, trainer: "Trainer", **kwargs) -> None:
        wandb = self._get_wandb()
        omegaconf = self._get_omegaconf()

        config = self.config
        if isinstance(config, dict):
            config = omegaconf.OmegaConf.create(config)
        settings = wandb.Settings(**self.kwargs)
        wandb.login()
        wandb.init(
            project=self.project_name,
            config=omegaconf.OmegaConf.to_container(config, resolve=True),
            dir=self.log_dir,
            settings=settings,
        )

    def on_train_step(self, trainer: "Trainer", step: int, logs: dict[str, Any], **kwargs) -> None:
        wandb = self._get_wandb()
        wandb.log({"train_step": step, **logs})

    def on_valid_step(
        self, trainer: "Trainer", step: int, val_id: str, predictions_dict: dict[str, Any], **kwargs
    ) -> None:
        wandb = self._get_wandb()
        wandb.log({f"val_{val_id}_{k}": v for k, v in predictions_dict.items()})

    def on_train_end(self, trainer: "Trainer", **kwargs) -> None:
        wandb = self._get_wandb()
        wandb.finish()


class TrainingCallbacks(BaseCallback):
    """Container for multiple callbacks."""

    def __init__(self, callbacks: Sequence[BaseCallback]) -> None:
        self.callbacks = callbacks
        # Dispatch by class. A callback may subclass both to receive both roles.
        self._computational = [cb for cb in callbacks if isinstance(cb, ComputationalCallback)]
        self._logging = [cb for cb in callbacks if isinstance(cb, LoggingCallback)]

        # Anything that is neither would silently never be called, so reject it here.
        unclassified = [cb for cb in callbacks if not isinstance(cb, ComputationalCallback | LoggingCallback)]
        if unclassified:
            names = ", ".join(sorted({type(cb).__name__ for cb in unclassified}))
            raise TypeError(
                f"Callbacks must subclass ComputationalCallback or LoggingCallback, got: {names}. "
                "Subclassing BaseCallback directly leaves the callback out of every dispatched hook."
            )

    def __len__(self) -> int:
        return len(self.callbacks)

    def on_train_begin(self, trainer: "Trainer", **kwargs) -> None:
        for cb in self.callbacks:
            cb.on_train_begin(trainer, **kwargs)

    def on_train_step(self, trainer: "Trainer", step: int, logs: dict[str, Any], **kwargs) -> None:
        for cb in self._logging:
            cb.on_train_step(trainer, step, logs, **kwargs)

    def on_valid_step(
        self, trainer: "Trainer", step: int, val_id: str, predictions_dict: dict[str, Any], **kwargs
    ) -> dict[str, Any]:
        metrics = {}
        for cb in self._computational:
            result = cb.on_valid_step(trainer, step, val_id, predictions_dict, **kwargs)
            if result:
                metrics.update(result)
        for cb in self._logging:
            cb.on_valid_step(trainer, step, val_id, metrics, **kwargs)
        return metrics

    def on_train_end(self, trainer: "Trainer", **kwargs) -> None:
        for cb in self.callbacks:
            cb.on_train_end(trainer, **kwargs)
