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
from scfoundations import GeneTokenizeCollate, GeneVocab
from scipy import sparse

__all__ = ["PanCellDataModule"]


def _dense(x) -> np.ndarray:
    return np.asarray(x.todense()) if sparse.issparse(x) else np.asarray(x)


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
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        # control & perturbed share var, so one tokenizer serves both; reusing scfoundations' collate is what
        # keeps this tokenization identical to how the GeneEncoder was trained (the warm-start precondition).
        self._tok = GeneTokenizeCollate(vocab.align(control_adata.var_names), max_tokens=max_tokens)
        self._length = steps_per_epoch * batch_size
        self._seed = seed

    def _collate(self, pairs):
        s = self._tok(np.stack([c for c, _ in pairs]))  # control cells -> source tokens
        t = self._tok(np.stack([p for _, p in pairs]))  # perturbed cells -> target tokens
        return {"source_tokens": s["tokens"], "source_mask": s["pad_mask"],
                "target_tokens": t["tokens"], "target_mask": t["pad_mask"]}

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        ds = _PairDataset(self._control, self._perturbed, self._length, self._seed)
        return torch.utils.data.DataLoader(ds, batch_size=self.batch_size, collate_fn=self._collate, num_workers=0)
