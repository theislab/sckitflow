"""One LightningModule harness + a registerable (model, objective) seam.

Training is always :class:`SCFlowLightningModule`; the weights are always a torch ``nn.Module``. What
varies — the loss math and whether it runs in torch or JAX — is an :class:`Objective`. Architectures and
objectives are registerable so third parties plug in their own without touching the framework. The base
seam lives here (ML core); concrete flow-matching objectives live in :mod:`sc_flow.flow` and register on
import.
"""

from sc_flow.core.training._harness import SCFlowLightningModule
from sc_flow.core.training._objective import (
    ARCHITECTURE_REGISTRY,
    OBJECTIVE_REGISTRY,
    Objective,
    build_architecture,
    build_objective,
    register_architecture,
    register_objective,
)

__all__ = [
    "SCFlowLightningModule",
    "Objective",
    "register_architecture",
    "register_objective",
    "build_architecture",
    "build_objective",
    "ARCHITECTURE_REGISTRY",
    "OBJECTIVE_REGISTRY",
]
