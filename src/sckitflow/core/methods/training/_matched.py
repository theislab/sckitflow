from __future__ import annotations

import torch
from scfit.registry import component

from sckitflow.core.coupling._config import CouplingConfig
from sckitflow.core.methods._base import MatchedTrainingMethod, Matcher, TrainingMethodConfig
from sckitflow.data._dims import DataDimensions

__all__ = ["MatchedTrainingConfig"]


@component("training_method.matched", builds=MatchedTrainingMethod)
class MatchedTrainingConfig(TrainingMethodConfig):
    """Pairs each batch with ``coupling``, then trains ``method`` on the pairs."""

    method: TrainingMethodConfig
    """The training method run on the matched batch, e.g. ``CFMTrainingConfig()``."""
    coupling: CouplingConfig
    """How source and target are paired, e.g. ``SinkhornCouplingConfig()``."""

    def build(self, module: torch.nn.Module, *, dims: DataDimensions) -> MatchedTrainingMethod:
        present = {
            "rep_lin": dims.source_lin_dim,
            "source_rep_quad": dims.source_quad_dim,
            "target_rep_quad": dims.target_quad_dim,
        }
        if missing := [f for f in self.coupling.reads if present[f] is None]:
            raise ValueError(
                f"{type(self.coupling).__name__} matches on {', '.join(self.coupling.reads)}, but the data manager "
                f"names no {', '.join(missing)}. Set it in `DataManagerConfig`."
            )
        matcher = Matcher(self.coupling.build(), needs_reps=bool(self.coupling.reads))
        return MatchedTrainingMethod(self.method.build(module, dims=dims), matcher)
