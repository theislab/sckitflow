"""The gene-set transformer backbone for contrastive cell pretraining (:mod:`sc_flow.concept`).

A cell is an ordered set of gene *tokens* (the genes it expresses, sorted by expression so position encodes
rank — see :mod:`sc_flow.concept._tokenize`). :class:`GeneEncoder` embeds the tokens, prepends a learnable
``CLS`` token, runs a pre-norm transformer, and returns the ``CLS`` output as the cell representation.

The encoder is a **pure, transferable backbone** — no contrastive machinery lives here. The learnable CLIP
temperature moved out to :class:`sc_flow.concept.ContrastiveHead`, so the same backbone loads unchanged for a
fine-tuning task (classification, decoding) that has no temperature. The returned vector is unnormalized;
the L2-normalized ``CLS`` is the contrastive space, applied by the objective / head.

:class:`GeneEncoderConfig` is a :class:`sc_flow.Component`: it round-trips as a ``{type, version, config}``
spec and builds the module. ``activation`` is a *string id* (``"gelu"``/``"relu"``) so the config stays
JSON-portable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from sc_flow.concept._vocab import PAD_TOKEN
from sc_flow.training._config import ArchitectureConfig

__all__ = ["GeneEncoder", "GeneEncoderConfig"]


def _sinusoidal_pe(max_len: int, dim: int) -> torch.Tensor:
    """Standard fixed sinusoidal positional encodings, shape ``(max_len, dim)``."""
    position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
    pe = torch.zeros(max_len, dim)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class GeneEncoder(nn.Module):
    """Embed a cell's gene-token set and return the ``CLS`` representation (unnormalized).

    Parameters mirror :class:`GeneEncoderConfig`. Attention uses PyTorch's built-in SDPA
    ``nn.TransformerEncoder``; a flash-attention fast path over unpadded sequences is a future drop-in.
    """

    def __init__(
        self,
        n_tokens: int,
        *,
        dim_model: int = 512,
        n_layers: int = 8,
        n_heads: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_rank: int = 5000,
        pad_token: int = PAD_TOKEN,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.dim_model = dim_model
        self.pad_token = pad_token
        self.gene_embedding = nn.Embedding(n_tokens, dim_model, padding_idx=pad_token)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim_model))
        self.input_dropout = nn.Dropout(dropout)
        self.register_buffer("pos_encoding", _sinusoidal_pe(max_rank, dim_model), persistent=False)
        layer = nn.TransformerEncoderLayer(
            d_model=dim_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,  # a string id ("gelu"/"relu") — keeps the config JSON-portable
            batch_first=True,
            norm_first=True,  # pre-norm, matches scConcept's `norm_scheme: pre`
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=n_layers, norm=nn.LayerNorm(dim_model), enable_nested_tensor=False
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.cls_token, std=0.02)
        with torch.no_grad():
            self.gene_embedding.weight[self.pad_token].zero_()

    def forward(self, tokens: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Encode one view. ``tokens`` ``(B, L)`` long; ``pad_mask`` ``(B, L)`` bool (``True`` = padded).

        Returns ``(B, dim_model)`` unnormalized ``CLS`` representation.
        """
        if tokens.dim() != 2:
            raise ValueError(f"tokens must be (batch, seq); got shape {tuple(tokens.shape)}.")
        batch, seq = tokens.shape
        if seq + 1 > self.pos_encoding.shape[0]:
            raise ValueError(f"sequence length {seq} (+CLS) exceeds max_rank {self.pos_encoding.shape[0]}.")

        x = self.gene_embedding(tokens)  # (B, L, D)
        cls = self.cls_token.expand(batch, -1, -1)  # (B, 1, D)
        x = torch.cat([cls, x], dim=1)  # (B, L+1, D) — position 0 is CLS, 1..L are gene ranks
        x = x + self.pos_encoding[: seq + 1].unsqueeze(0)
        x = self.input_dropout(x)

        key_padding_mask = None
        if pad_mask is not None:
            cls_keep = torch.zeros(batch, 1, dtype=torch.bool, device=tokens.device)  # CLS never padded
            key_padding_mask = torch.cat([cls_keep, pad_mask], dim=1)  # (B, L+1), True = ignore

        hidden = self.transformer(x, src_key_padding_mask=key_padding_mask)  # (B, L+1, D)
        return hidden[:, 0]  # CLS representation (already layer-normed)


@dataclass
class GeneEncoderConfig(ArchitectureConfig, type_id="sc_flow.gene_encoder", version=1):
    """Portable recipe for :class:`GeneEncoder` (a :class:`sc_flow.Component`).

    ``n_tokens`` is the embedding-table size — take it from :attr:`GeneVocab.n_tokens`. ``max_rank`` bounds
    the number of gene positions (``CLS`` occupies position 0). ``activation`` is a string id.
    """

    n_tokens: int
    dim_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    dim_feedforward: int = 1024
    dropout: float = 0.1
    max_rank: int = 5000
    pad_token: int = PAD_TOKEN
    activation: str = "gelu"

    def build(self, context: object = None) -> GeneEncoder:
        return GeneEncoder(
            n_tokens=self.n_tokens,
            dim_model=self.dim_model,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            max_rank=self.max_rank,
            pad_token=self.pad_token,
            activation=self.activation,
        )
