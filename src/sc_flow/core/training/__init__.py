"""One LightningModule harness + registerable (model, objective, predictor) seams.

Training is always :class:`TrainingModule`; the weights are always a torch ``nn.Module``. What varies —
the loss math (torch or JAX) and how a trained model predicts — are the :class:`Objective` (train seam)
and :class:`Predictor` (inference seam). Architectures, objectives and predictors are registerable so
third parties plug in their own without touching the framework. The base seams live here (ML core);
concrete flow-matching implementations live in :mod:`sc_flow.flow` and register on import.
"""

from sc_flow.core.training._harness import TrainingModule
from sc_flow.core.training._objective import (
    ARCHITECTURE_REGISTRY,
    OBJECTIVE_REGISTRY,
    Objective,
    build_architecture,
    build_objective,
    register_architecture,
    register_objective,
)
from sc_flow.core.training._predictor import (
    PREDICTOR_REGISTRY,
    Predictor,
    build_predictor,
    register_predictor,
)

__all__ = [
    "TrainingModule",
    "Objective",
    "Predictor",
    "register_architecture",
    "register_objective",
    "register_predictor",
    "build_architecture",
    "build_objective",
    "build_predictor",
    "ARCHITECTURE_REGISTRY",
    "OBJECTIVE_REGISTRY",
    "PREDICTOR_REGISTRY",
]
