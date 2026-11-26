from __future__ import annotations

import torch
from modelbinder.registry import component

from sckitflow.core.coupling._config import CouplingConfig
from sckitflow.core.methods._base import MatchedTrainingMethod, Matcher, TrainingMethodConfig
from sckitflow.data._dims import DataDimensions

__all__ = ["MatchedTrainingConfig"]


@component("training_method.matched")
class MatchedTrainingConfig(TrainingMethodConfig):
    """Pairs each batch with ``coupling``, then trains ``method`` on the pairs."""

    method: TrainingMethodConfig
    """The training method run on the matched batch, e.g. ``CFMTrainingConfig()``."""
    coupling: CouplingConfig
    """How source and target are paired, e.g. ``SinkhornCouplingConfig()``."""

    def build(self, module: torch.nn.Module, *, dims: DataDimensions) -> MatchedTrainingMethod:
        return MatchedTrainingMethod(self.method.build(module, dims=dims), Matcher(self.coupling.build()))
