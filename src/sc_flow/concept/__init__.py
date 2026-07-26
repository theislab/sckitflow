"""Contrastive cell pretraining (:mod:`sc_flow.concept`) — a CLIP-style single-cell foundation model.

A second model family alongside :mod:`sc_flow.flow`, built on the family-neutral core
(:mod:`sc_flow.training`, :mod:`sc_flow._registry`) plus a single-cell data path. Two gene-panel *views* of
the same cell are trained to embed together (InfoNCE); other cells in the batch are negatives — no
reconstruction, no perturbation pairing. Reimplemented natively — **no dependency on scConcept or
lamin-dataloader** (the method is Bahrami et al. 2025, ``theislab/scConcept``).

Layering: a pure transferable **backbone** (:class:`GeneEncoder`) + a task **head** (:class:`ContrastiveHead`)
composed into a trainable model, assembled by the **family builder** :class:`FoundationModel` (peer to
:class:`sc_flow.FlowMatching`). Architectures + objectives are :class:`sc_flow.Component`\\s (portable specs).

``import sc_flow.concept`` pulls torch + the neutral training core (lightning, via the objective's registry
import) + cattrs — but never the flow stack (jax/ott/torchdiffeq). The anndata/scipy data path and the
:class:`FoundationModel` builder are imported lazily so the base import stays light.
"""

from __future__ import annotations

from sc_flow.concept._encoder import GeneEncoder, GeneEncoderConfig
from sc_flow.concept._head import ContrastiveHead, ContrastiveModel
from sc_flow.concept._objective import ContrastiveObjective, ContrastiveObjectiveConfig, ObjectiveConfig
from sc_flow.concept._tokenize import TwoViewCollate, rank_encode
from sc_flow.concept._vocab import CLS_TOKEN, NUM_SPECIAL, PAD_TOKEN, GeneVocab

__all__ = [
    "GeneVocab",
    "PAD_TOKEN",
    "CLS_TOKEN",
    "NUM_SPECIAL",
    "TwoViewCollate",
    "rank_encode",
    "GeneEncoder",
    "GeneEncoderConfig",
    "ContrastiveHead",
    "ContrastiveModel",
    "ContrastiveObjective",
    "ObjectiveConfig",
    "ContrastiveObjectiveConfig",
    "FoundationModel",
    "FoundationDataModule",
]


def __getattr__(name: str):
    # The builder + datamodule pull lightning eagerly; keep the base import torch-only by loading them lazily.
    if name == "FoundationModel":
        from sc_flow.concept._builder import FoundationModel

        return FoundationModel
    if name == "FoundationDataModule":
        from sc_flow.concept._data import FoundationDataModule

        return FoundationDataModule
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
