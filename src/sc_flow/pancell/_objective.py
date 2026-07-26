"""The flow-matching objective as a :class:`sc_flow.Component` — symmetric with the contrastive one.

``LinearFMObjectiveConfig`` is an :class:`sc_flow.training.ObjectiveConfig` that nests a
``probability_path`` Component (exactly the shape ``OTFMObjectiveConfig`` takes, minus the jax/ott coupling
— this variant is torch-only so the composition runs anywhere). Its ``compute_loss`` transports
source→target **in the encoder's latent**: it asks the model to encode both populations, then matches the
velocity to the path's target vector field. That the encoder is shared is what makes cross-dataset transport
well-defined.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from sc_flow._registry import Component
from sc_flow.training._config import ObjectiveConfig
from sc_flow.training._objective import Objective

__all__ = ["ProbabilityPath", "LinearPathConfig", "LinearFMObjective", "LinearFMObjectiveConfig"]


class LinearPath:
    """Rectified-flow path: ``x_t = (1-t) x0 + t x1``, target velocity ``u = x1 - x0`` (optional noise)."""

    def __init__(self, sigma: float = 0.0) -> None:
        self.sigma = sigma

    def xt(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        xt = (1.0 - t) * x0 + t * x1
        return xt if self.sigma == 0.0 else xt + self.sigma * torch.randn_like(xt)

    def ut(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return x1 - x0


class ProbabilityPath(Component):
    """Abstract family base for probability-path configs (unregistered). ``build(ctx) -> a runtime path``."""

    def build(self, context: Any = None) -> Any:
        raise NotImplementedError


@dataclass
class LinearPathConfig(ProbabilityPath, type_id="flow.path.linear", version=1):
    sigma: float = 0.0

    def build(self, context: Any = None) -> LinearPath:
        return LinearPath(self.sigma)


class LinearFMObjective(Objective):
    def __init__(self, path: LinearPath) -> None:
        self._path = path

    def compute_loss(self, model: torch.nn.Module, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        z0 = model.encode(batch["source_tokens"], batch.get("source_mask"))  # control cells → latent
        z1 = model.encode(batch["target_tokens"], batch.get("target_mask"))  # perturbed cells → latent
        t = torch.rand(z0.shape[0], 1, device=z0.device, dtype=z0.dtype)
        z_t = self._path.xt(z0, z1, t)
        u = self._path.ut(z0, z1, t)
        v = model.velocity(z_t, t)
        loss = ((v - u) ** 2).mean()
        return loss, {"loss": loss.detach(), "fm_loss": loss.detach(),
                      "latent_norm": z0.detach().norm(dim=-1).mean()}


@dataclass
class LinearFMObjectiveConfig(ObjectiveConfig, type_id="flow.fm_linear", version=1):
    probability_path: ProbabilityPath = field(default_factory=LinearPathConfig)  # nested Component

    def build(self, context: Any = None) -> LinearFMObjective:
        return LinearFMObjective(self.probability_path.build(context))
