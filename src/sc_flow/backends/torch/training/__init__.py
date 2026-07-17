"""One LightningModule harness + a registerable (model, objective) seam.

Training is always :class:`SCFlowLightningModule`; the weights are always a torch
``nn.Module``. What varies — the loss math and whether it runs in torch or JAX — is an
:class:`Objective`. Architectures and objectives are registerable so third parties can
plug in their own without touching the framework. See :mod:`._objective` for the seam.
"""

from sc_flow.backends.torch.training._harness import SCFlowLightningModule
from sc_flow.backends.torch.training._objective import (
    ARCHITECTURE_REGISTRY,
    OBJECTIVE_REGISTRY,
    Objective,
    TorchLinearFMObjective,
    build_architecture,
    build_objective,
    register_architecture,
    register_objective,
)

__all__ = [
    "SCFlowLightningModule",
    "Objective",
    "TorchLinearFMObjective",
    "register_architecture",
    "register_objective",
    "build_architecture",
    "build_objective",
    "ARCHITECTURE_REGISTRY",
    "OBJECTIVE_REGISTRY",
]
