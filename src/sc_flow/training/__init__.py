"""Back-compat shim: the training core moved to :mod:`scfit.training`. Import from there in new code."""

from scfit.training import (
    OBJECTIVE_REGISTRY,
    ArchitectureConfig,
    Objective,
    ObjectiveConfig,
    Predictor,
    TrainingModule,
    build_objective,
    register_objective,
)

__all__ = [
    "TrainingModule",
    "Objective",
    "Predictor",
    "ArchitectureConfig",
    "ObjectiveConfig",
    "register_objective",
    "build_objective",
    "OBJECTIVE_REGISTRY",
]
