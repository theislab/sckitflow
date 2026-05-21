from typing import Any

import torch
import torch.nn as nn


class ExponentialMovingAverage:
    """Exponential Moving Average for model parameters.

    Maintains shadow copies of model parameters and updates them using EMA.
    Generative models (diffusion, flow matching, etc.) are trained with EMA.

    Args:
        model: The model whose parameters to track
        decay: EMA decay rate (typically 0.999 or 0.9999)
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
    ):
        self._model = model
        self._decay = decay

        # Create shadow parameters
        self._shadow_params = {}
        self._register_parameters()

    def _register_parameters(self) -> None:
        """Register model parameters for EMA tracking."""
        for name, param in self._model.named_parameters():
            if param.requires_grad:
                self._shadow_params[name] = param.data.clone().detach()

    def update(self) -> None:
        """Update EMA parameters."""
        with torch.no_grad():
            for name, param in self._model.named_parameters():
                if param.requires_grad and name in self._shadow_params:
                    self._shadow_params[name].mul_(self._decay).add_(param.data, alpha=1 - self._decay)

    def copy_to(self, model: nn.Module) -> None:
        """Copy EMA parameters to a model."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self._shadow_params:
                    param.data.copy_(self._shadow_params[name])

    def store(self) -> dict[str, torch.Tensor]:
        """Store current model parameters (for restoration later)."""
        stored_params = {}
        for name, param in self._model.named_parameters():
            if param.requires_grad:
                stored_params[name] = param.data.clone().detach()
        return stored_params

    def restore(self, stored_params: dict[str, torch.Tensor]) -> None:
        """Restore model parameters from stored values."""
        with torch.no_grad():
            for name, param in self._model.named_parameters():
                if param.requires_grad and name in stored_params:
                    param.data.copy_(stored_params[name])

    def state_dict(self) -> dict[str, Any]:
        """Return state dict for checkpointing."""
        return {
            "shadow_params": self._shadow_params,
            "decay": self._decay,
        }

    def load_state_dict(self, state_dict: dict[str, Any], strict: bool = True) -> None:
        """Load state dict with optional strict key checking."""
        if strict and set(state_dict["shadow_params"].keys()) != set(self._shadow_params.keys()):
            raise RuntimeError("Shadow param keys mismatch")
        self._shadow_params = state_dict["shadow_params"]
        self._decay = state_dict["decay"]

    def to(self, device: torch.device) -> "ExponentialMovingAverage":
        """Move EMA to device."""
        for name in self._shadow_params:
            self._shadow_params[name] = self._shadow_params[name].to(device)
        return self

    def __enter__(self):
        """Context manager to temporarily use EMA weights."""
        self._stored_params = self.store()
        self.copy_to(self._model)
        return self

    def __exit__(self, *args):
        """Restore original weights when exiting context."""
        self.restore(self._stored_params)
        del self._stored_params
