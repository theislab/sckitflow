"""The contrastive (CLIP / InfoNCE) training objective for :mod:`sc_flow.concept`.

Two gene-panel *views* of the same batch of cells are encoded to L2-normalized embeddings; the symmetric
cross-entropy pulls each cell's two views together while pushing every other cell in the batch apart (the
other cells are the negatives). This is the whole pretraining signal — no reconstruction, no perturbation
pairing.

Registered as ``"concept-clip"`` in :data:`sc_flow.training.OBJECTIVE_REGISTRY`, so it drops into the
family-neutral :class:`sc_flow.training.TrainingModule` exactly like the flow objectives. The model handed
to :meth:`compute_loss` must encode a view (``model(tokens, pad_mask) -> (batch, dim)``) and expose a
learnable ``logit_scale`` — :class:`sc_flow.concept.GeneEncoder` satisfies both.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from sc_flow._registry import Component
from sc_flow.training._objective import Objective, register_objective

__all__ = ["ContrastiveObjective", "ObjectiveConfig", "ContrastiveObjectiveConfig"]


@register_objective("concept-clip")
class ContrastiveObjective(Objective):
    """Symmetric InfoNCE over two views. Batch keys: ``tokens_1/2`` (long ``(B, L)``) + optional
    ``pad_mask_1/2`` (bool ``(B, L)``, ``True`` = padded).

    Parameters
    ----------
    max_logit_scale
        Upper clamp on the temperature ``exp(logit_scale)`` (CLIP uses 100) — guards against the
        similarities blowing up early in training. The clamp is applied at use and never mutates the
        parameter.
    """

    def __init__(self, *, max_logit_scale: float = 100.0) -> None:
        self._max_log_scale = math.log(max_logit_scale)

    def compute_loss(self, model: torch.nn.Module, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        z1 = F.normalize(model(batch["tokens_1"], batch.get("pad_mask_1")), dim=-1)  # (B, D)
        z2 = F.normalize(model(batch["tokens_2"], batch.get("pad_mask_2")), dim=-1)  # (B, D)

        scale = model.logit_scale.clamp(max=self._max_log_scale).exp()
        logits = scale * (z1 @ z2.t())  # (B, B): row i = view-1 of cell i vs view-2 of every cell
        labels = torch.arange(z1.shape[0], device=z1.device)

        # Symmetric: predict the matching view-2 from each view-1, and vice versa.
        loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))

        with torch.no_grad():
            acc = (logits.argmax(dim=1) == labels).float().mean()  # in-batch retrieval accuracy
        return loss, {"loss": loss.detach(), "logit_scale": scale.detach(), "retrieval_acc": acc}


class ObjectiveConfig(Component):
    """Abstract family base for objective *configs* (unregistered). ``build(ctx)`` returns the runtime
    :class:`sc_flow.training.Objective` that owns ``compute_loss``."""


@dataclass
class ContrastiveObjectiveConfig(ObjectiveConfig, type_id="sc_flow.concept_clip", version=1):
    """Portable recipe for the contrastive task (a :class:`sc_flow.Component`).

    Carries ``logit_scale_init`` (the :class:`sc_flow.concept.ContrastiveHead` temperature the builder wires
    onto the model) and ``max_logit_scale`` (the loss-time clamp). ``build`` returns the runtime objective.
    """

    logit_scale_init: float = 3.0
    max_logit_scale: float = 100.0

    def build(self, context: object = None) -> ContrastiveObjective:
        return ContrastiveObjective(max_logit_scale=self.max_logit_scale)
