"""Rank encoding + the two-view contrastive augmentation (:mod:`sc_flow.concept`).

Turns a batch of raw-count cells into the two disjoint gene-panel *views* the contrastive objective
consumes. Source-agnostic: it takes a count matrix ``(batch, n_vars)`` plus a ``var -> token`` alignment
(:meth:`sc_flow.concept.GeneVocab.align`), so it works over an in-memory AnnData batch or a streamed
:mod:`scfit.data` batch alike.

Per cell: take the expressed, in-vocabulary genes; randomly split them into two **disjoint** panels; within
each panel keep the top-``max_tokens`` genes by expression, ordered by descending count (rank encoding —
position carries magnitude, the values themselves are not fed to the model). The two views share no genes,
which is what teaches panel/technology invariance.

Tokenization is host-side (numpy). For a GPU stream this means one device→host hop per batch in the collate;
fine for Phase 1, and the natural place to later vectorize or push into the dataset worker.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from sc_flow.concept._vocab import PAD_TOKEN

__all__ = ["TwoViewCollate", "rank_encode"]


def _as_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if hasattr(x, "toarray"):  # scipy sparse
        return np.asarray(x.toarray())
    return np.asarray(x)


def rank_encode(tokens: np.ndarray, counts: np.ndarray, max_tokens: int) -> np.ndarray:
    """Order ``tokens`` by descending ``counts``, keep the top ``max_tokens``. Returns 1D int64 tokens."""
    if tokens.size == 0:
        return tokens.astype(np.int64, copy=False)
    order = np.argsort(-counts, kind="stable")  # highest expression first; stable for reproducible ties
    return tokens[order][:max_tokens].astype(np.int64, copy=False)


def _pad(rows: list[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
    """Left-pack variable-length token rows into ``(B, L)`` + a bool pad mask (``True`` = padded)."""
    length = max((len(r) for r in rows), default=0)
    length = max(length, 1)  # keep at least one column so a batch of empty cells still forms a tensor
    tokens = np.full((len(rows), length), PAD_TOKEN, dtype=np.int64)
    pad_mask = np.ones((len(rows), length), dtype=bool)
    for i, r in enumerate(rows):
        n = len(r)
        if n:
            tokens[i, :n] = r
            pad_mask[i, :n] = False
    return torch.from_numpy(tokens), torch.from_numpy(pad_mask)


@dataclass
class TwoViewCollate:
    """Collate a raw-count batch into two disjoint, rank-encoded gene-panel views.

    Parameters
    ----------
    var_token
        int64 array ``(n_vars,)`` from :meth:`GeneVocab.align`: column -> token, ``-1`` if unmapped.
    max_tokens
        Cap on genes kept per panel (the sequence length before padding).
    seed
        Base seed for the per-call gene shuffle/split; the generator advances each call (reproducible run,
        different split every batch).

    Returns (from ``__call__``) a batch dict with ``tokens_1/2`` (long ``(B, L)``) and ``pad_mask_1/2``
    (bool ``(B, L)``, ``True`` = padded) — exactly what :class:`sc_flow.concept.ContrastiveObjective` and
    :class:`sc_flow.concept.GeneEncoder` consume.
    """

    var_token: np.ndarray
    max_tokens: int = 1024
    seed: int = 0
    _rng: np.random.Generator = field(init=False, repr=False)
    _valid: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.var_token = np.asarray(self.var_token, dtype=np.int64)
        if self.var_token.ndim != 1:
            raise ValueError(f"var_token must be 1D (n_vars,); got shape {self.var_token.shape}.")
        self._rng = np.random.default_rng(self.seed)
        self._valid = self.var_token >= 0  # in-vocabulary columns

    def __call__(self, x) -> dict[str, torch.Tensor]:
        counts = _as_numpy(x)
        if counts.ndim != 2:
            raise ValueError(f"expected a (batch, n_vars) count matrix; got shape {counts.shape}.")
        if counts.shape[1] != self.var_token.shape[0]:
            raise ValueError(f"n_vars mismatch: counts has {counts.shape[1]}, var_token has {self.var_token.shape[0]}.")

        rows_1: list[np.ndarray] = []
        rows_2: list[np.ndarray] = []
        for row in counts:
            expressed = np.nonzero((row > 0) & self._valid)[0]  # in-vocab genes this cell expresses
            self._rng.shuffle(expressed)
            half = len(expressed) // 2
            panel_1, panel_2 = expressed[:half], expressed[half:]  # disjoint; panel_2 keeps the odd gene
            rows_1.append(rank_encode(self.var_token[panel_1], row[panel_1], self.max_tokens))
            rows_2.append(rank_encode(self.var_token[panel_2], row[panel_2], self.max_tokens))

        tokens_1, pad_mask_1 = _pad(rows_1)
        tokens_2, pad_mask_2 = _pad(rows_2)
        return {"tokens_1": tokens_1, "tokens_2": tokens_2, "pad_mask_1": pad_mask_1, "pad_mask_2": pad_mask_2}
