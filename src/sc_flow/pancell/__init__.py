"""Pan-cell flow — flow matching in a foundation encoder's shared latent (cross-dataset composition).

Composes the two families: a :class:`sc_flow.concept.GeneEncoder` (from the foundation toolbox) sits in the
**state-encoder slot** of a flow model, so cells from *different gene panels* map into one latent and a
rectified-flow velocity transports source→target across datasets — frozen or fine-tuned. Importing this
package registers the ``"pancell"`` and ``"foundation"`` families.
"""

from __future__ import annotations

from sc_flow.pancell._builder import FoundationFamily, PanCellFlow, PanCellFlowFamily
from sc_flow.pancell._model import PanCellFlowModel, VelocityMLP, VelocityMLPConfig
from sc_flow.pancell._objective import (
    LinearFMObjective,
    LinearFMObjectiveConfig,
    LinearPathConfig,
    ProbabilityPath,
)

__all__ = [
    "PanCellFlow",
    "PanCellFlowFamily",
    "FoundationFamily",
    "PanCellFlowModel",
    "VelocityMLP",
    "VelocityMLPConfig",
    "LinearFMObjective",
    "LinearFMObjectiveConfig",
    "LinearPathConfig",
    "ProbabilityPath",
]
