"""Single-cell data plumbing for contrastive pretraining (:mod:`sc_flow.concept`).

A :class:`FoundationDataModule` turns an in-memory :class:`~anndata.AnnData` (raw counts, Ensembl ``var``)
into the two-view token batches the objective consumes, via :class:`sc_flow.concept.TwoViewCollate`. This is
the in-memory path used for a single dataset (or a few concatenated) — enough for a real end-to-end run.
Streaming the full multi-species CellxGene corpus (thousands of shards, ~TBs) via ``scfit.data`` /
converted zarr is the documented scale-up; the collate is source-agnostic, so only the loader changes.
"""

from __future__ import annotations

import lightning.pytorch as pl
import numpy as np
import torch
from scipy import sparse

from sc_flow.concept._tokenize import TwoViewCollate
from sc_flow.concept._vocab import GeneVocab

__all__ = ["AnnDataCellDataset", "FoundationDataModule"]


class AnnDataCellDataset(torch.utils.data.Dataset):
    """Yield a cell's dense raw-count row; the two-view collate does tokenization + augmentation."""

    def __init__(self, adata) -> None:
        self._x = adata.X

    def __len__(self) -> int:
        return self._x.shape[0]

    def __getitem__(self, i: int) -> np.ndarray:
        row = self._x[i]
        if sparse.issparse(row):
            return np.asarray(row.todense()).ravel().astype(np.float32)
        return np.asarray(row, dtype=np.float32).ravel()


class FoundationDataModule(pl.LightningDataModule):
    """Stream single cells from an AnnData as two disjoint gene-panel views.

    Parameters
    ----------
    adata
        Raw-count :class:`~anndata.AnnData` with Ensembl ``var_names``.
    vocab
        The :class:`GeneVocab` defining the token space (built from the corpus / this dataset).
    batch_size, max_tokens, seed
        Batch size, per-view gene cap, and the collate's base seed.
    num_workers
        DataLoader workers. Default 0 (main process) — host-side tokenization is cheap and keeps the
        ``lambda`` collate simple; ``> 0`` needs a picklable collate (a small follow-up).
    """

    def __init__(
        self,
        adata,
        vocab: GeneVocab,
        *,
        batch_size: int = 256,
        max_tokens: int = 1024,
        seed: int = 0,
        num_workers: int = 0,
    ) -> None:
        super().__init__()
        self._ds = AnnDataCellDataset(adata)
        self._collate = TwoViewCollate(vocab.align(adata.var_names), max_tokens=max_tokens, seed=seed)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self._ds,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
            collate_fn=lambda samples: self._collate(np.stack(samples)),
        )
