from __future__ import annotations

from collections.abc import Callable
from functools import partial

import torch
from scfit.registry import component

from sckitflow.core.nn.fields._base import FIELD_DECODER_KEY, BaseNeuralFieldConfig, NeuralFieldMixin
from sckitflow.data._dims import DataDimensions

__all__ = ["NeuralVelocity", "NeuralVelocityConfig"]


class NeuralVelocity(NeuralFieldMixin):
    """Time-conditional field: ``v(t, x) -> state_dim``."""

    def forward(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        *,
        condition_dict: dict[str, torch.Tensor] | None = None,
        source: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._field[FIELD_DECODER_KEY](self._encode(t, x, cond_dict=condition_dict, source=source))

    def get_vf_fn(
        self,
        condition_dict: dict[str, torch.Tensor] | None = None,
        source: torch.Tensor | None = None,
    ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Binds the condition and source, returning ``v(t, x)`` for the ODE/SDE solvers."""
        return partial(self.forward, condition_dict=condition_dict, source=source)


@component("nn.models.velocity", builds=NeuralVelocity)
class NeuralVelocityConfig(BaseNeuralFieldConfig):
    def build(self, dims: DataDimensions) -> NeuralVelocity:
        return NeuralVelocity(self, dims=dims)
