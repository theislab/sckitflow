from __future__ import annotations

from functools import partial
from typing import Literal

import torch
from scfit.registry import component

from sckitflow.core.coupling import ot_linear_coupling
from sckitflow.core.methods._base import MatchedTrainingMethod, Matcher, TrainingMethodConfig

__all__ = ["OTMatchedTrainingConfig"]


@component("training_method.ot_matched", builds=MatchedTrainingMethod)
class OTMatchedTrainingConfig(TrainingMethodConfig):
    """Pairs each batch by linear optimal transport, then trains ``method`` on the pairs."""

    method: TrainingMethodConfig
    """The training method run on the matched batch, e.g. ``CFMTrainingConfig()``."""
    solver: Literal["exact", "sinkhorn", "partial", "unbalanced"] = "sinkhorn"
    reg: float = 0.5
    reg_m: float = 1.0
    """Marginal relaxation, used only by ``"unbalanced"``."""
    scale_cost: Literal["mean", "max", "median"] | float = "mean"

    def build(self, module: torch.nn.Module) -> MatchedTrainingMethod:
        match_fn = partial(
            ot_linear_coupling, method=self.solver, reg=self.reg, reg_m=self.reg_m, scale_cost=self.scale_cost
        )
        return MatchedTrainingMethod(self.method.build(module), Matcher(match_fn))
