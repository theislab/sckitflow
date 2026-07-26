"""Task heads + the composed task-model for :mod:`sc_flow.concept`.

The **backbone** (:class:`sc_flow.concept.GeneEncoder`) is a pure, transferable representation. A *task*
attaches a small **head** and composes the two into the trainable model the objective drives. This is the
seam that makes fine-tuning clean: pretraining uses a :class:`ContrastiveHead` (just the CLIP temperature),
a downstream task swaps in its own head (e.g. a classification linear) over the *same* backbone.

Phase-1 ships the contrastive (pretrain) task. ``ClassificationHead`` / ``ExpressionDecoderHead`` are the
documented fine-tuning extension points — same backbone, different head + objective.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["ContrastiveHead", "ContrastiveModel"]


class ContrastiveHead(nn.Module):
    """The pretraining head: the learnable CLIP temperature (log space).

    Kept off the backbone so the backbone stays a pure representation. Lives on the model so the neutral
    :class:`sc_flow.training.TrainingModule` (which optimizes ``model.parameters()``) trains it.
    """

    def __init__(self, logit_scale_init: float = 3.0) -> None:
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor(float(logit_scale_init)))


class ContrastiveModel(nn.Module):
    """The trainable contrastive model = backbone + :class:`ContrastiveHead`.

    Exposes exactly what :class:`sc_flow.concept.ContrastiveObjective` consumes: ``forward(tokens, pad_mask)``
    returns the (unnormalized) cell representation, and ``.logit_scale`` delegates to the head. For plain
    embedding extraction, use ``model.backbone`` directly.
    """

    def __init__(self, backbone: nn.Module, head: ContrastiveHead) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    @property
    def logit_scale(self) -> torch.Tensor:
        return self.head.logit_scale

    def forward(self, tokens: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.backbone(tokens, pad_mask)
