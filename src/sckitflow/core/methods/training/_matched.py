from __future__ import annotations

import torch
from modelbinder.registry import component

from sckitflow.core.coupling._config import (
    CouplingConfig,
    FusedGromovWassersteinCouplingConfig,
    IndependentCouplingConfig,
    _LinearCouplingConfig,
    _QuadraticCouplingConfig,
)
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
        matcher = Matcher(self.coupling.build(), needs_reps=_check_coupling_reps(self.coupling, dims))
        return MatchedTrainingMethod(self.method.build(module, dims=dims), matcher)


def _check_coupling_reps(coupling: CouplingConfig, dims: DataDimensions) -> bool:
    """Raise if the data manager names none of the reps ``coupling`` matches on; return whether it matches on any.

    Temporary, see theislab/sckitflow#237: once a coupling names its own reps as references and the model collects
    the required fields of its data manager and its coupling, this and ``Matcher(needs_reps=...)`` go.
    """
    if isinstance(coupling, IndependentCouplingConfig):
        return False  # random pairs read only the row counts
    present = {}
    if isinstance(coupling, _LinearCouplingConfig | FusedGromovWassersteinCouplingConfig):
        present["rep_lin"] = dims.source_lin_dim
    if isinstance(coupling, _QuadraticCouplingConfig):
        present |= {"source_rep_quad": dims.source_quad_dim, "target_rep_quad": dims.target_quad_dim}
    if missing := [field for field, dim in present.items() if dim is None]:
        raise ValueError(
            f"{type(coupling).__name__} matches on {', '.join(present)}, but the data manager names no "
            f"{', '.join(missing)}. Set it in `DataManagerConfig`."
        )
    return True
