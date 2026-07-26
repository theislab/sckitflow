"""Data path for pan-cell flow: control/perturbed populations → source/target token batches.

Single-view rank tokenization (the *whole* cell, unlike the contrastive two-view split), over a shared
:class:`scfoundations.GeneVocab`. Control and perturbed cells may come from different datasets / panels —
they share the vocab, so the encoder maps them into one latent. Independent-coupling sampling (a random
control paired with a random perturbed) is enough for rectified flow.
"""

from __future__ import annotations

import lightning.pytorch as pl
import numpy as np
import torch
from scipy import sparse

from scfoundations._tokenize import rank_encode
from scfoundations._vocab import PAD_TOKEN, GeneVocab

__all__ = ["tokenize_batch", "PanCellDataModule"]


def _dense(x) -> np.ndarray:
    return np.asarray(x.todense()) if sparse.issparse(x) else np.asarray(x)


def tokenize_batch(rows: np.ndarray, var_token: np.ndarray, valid: np.ndarray, max_tokens: int):
    """Rank-tokenize each cell's expressed, in-vocab genes; left-pack to ``(B, L)`` + a bool pad mask."""
    toks = []
    for row in rows:
        expressed = np.nonzero((row > 0) & valid)[0]
        toks.append(rank_encode(var_token[expressed], row[expressed], max_tokens))
    length = max(max((len(t) for t in toks), default=1), 1)
    out = np.full((len(toks), length), PAD_TOKEN, dtype=np.int64)
    mask = np.ones((len(toks), length), dtype=bool)
    for i, t in enumerate(toks):
        if len(t):
            out[i, : len(t)] = t
            mask[i, : len(t)] = False
    return torch.from_numpy(out), torch.from_numpy(mask)


class _PairDataset(torch.utils.data.Dataset):
    """Length = steps × batch; each item pairs a random control row with a random perturbed row."""

    def __init__(self, control: np.ndarray, perturbed: np.ndarray, length: int, seed: int) -> None:
        self._c, self._p, self._n = control, perturbed, length
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, _i: int):
        return self._c[self._rng.integers(len(self._c))], self._p[self._rng.integers(len(self._p))]


class PanCellDataModule(pl.LightningDataModule):
    def __init__(self, control_adata, perturbed_adata, vocab: GeneVocab, *, batch_size: int = 128,
                 max_tokens: int = 256, steps_per_epoch: int = 2000, seed: int = 0) -> None:
        super().__init__()
        self._control = _dense(control_adata.X)
        self._perturbed = _dense(perturbed_adata.X)
        self._var_token = vocab.align(control_adata.var_names)  # control & perturbed share var
        self._valid = self._var_token >= 0
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self._length = steps_per_epoch * batch_size
        self._seed = seed

    def _collate(self, pairs):
        ctrl = np.stack([c for c, _ in pairs])
        pert = np.stack([p for _, p in pairs])
        st, sm = tokenize_batch(ctrl, self._var_token, self._valid, self.max_tokens)
        tt, tm = tokenize_batch(pert, self._var_token, self._valid, self.max_tokens)
        return {"source_tokens": st, "source_mask": sm, "target_tokens": tt, "target_mask": tm}

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        ds = _PairDataset(self._control, self._perturbed, self._length, self._seed)
        return torch.utils.data.DataLoader(ds, batch_size=self.batch_size, collate_fn=self._collate, num_workers=0)
