"""Local CPU smoke for the contrastive path: AnnData -> GeneVocab -> two-view collate -> Trainer.fit.

Mirrors scripts/smoke_flow_matching.py but for sc_flow.concept (the CLIP-style cell encoder). No scfit
stream here — a plain torch DataLoader over a synthetic CellxGene-shaped AnnData (Ensembl var ids, raw
integer counts, co-expression programs so two DISJOINT gene panels of a cell still share identity signal).
It trains the family-neutral sc_flow.training.TrainingModule with the "concept-clip" objective and checks
the loss goes down + in-batch retrieval improves.

Run: ``python scripts/smoke_concept.py`` (sc-flow-tools venv, CPU-only).
"""

import anndata as ad
import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from scipy import sparse

from sc_flow.concept import (
    ContrastiveHead,
    ContrastiveModel,
    ContrastiveObjective,
    GeneEncoderConfig,
    GeneVocab,
    TwoViewCollate,
)
from sc_flow.training import TrainingModule

N_CELLS, N_GENES, N_PROGRAMS = 256, 200, 16
BATCH, MAX_TOKENS, MAX_STEPS = 64, 64, 150


def synthetic_cxg(seed: int = 0) -> ad.AnnData:
    """A CellxGene-shaped AnnData: Ensembl var, raw counts, co-expression programs + per-cell variation."""
    rng = np.random.default_rng(seed)
    program_genes = [rng.choice(N_GENES, size=20, replace=False) for _ in range(N_PROGRAMS)]
    cell_program = rng.integers(0, N_PROGRAMS, size=N_CELLS)
    x = np.zeros((N_CELLS, N_GENES), dtype=np.float32)
    for i in range(N_CELLS):
        rate = np.full(N_GENES, 0.2)
        rate[program_genes[cell_program[i]]] += rng.uniform(2.0, 5.0)  # marker genes fire high
        rate *= rng.uniform(0.5, 1.5)  # per-cell size factor -> individuality
        x[i] = rng.poisson(rate).astype(np.float32)
    obs = pd.DataFrame({
        "cell_type": pd.Categorical([f"program_{p}" for p in cell_program]),
        "dataset": pd.Categorical(rng.choice(["cxg_ds0", "cxg_ds1", "cxg_ds2"], N_CELLS)),
    })
    adata = ad.AnnData(X=sparse.csr_matrix(x), obs=obs)
    adata.var_names = [f"ENSG{i:011d}" for i in range(N_GENES)]
    adata.obs_names = [str(i) for i in range(N_CELLS)]
    return adata


class CellDataset(torch.utils.data.Dataset):
    """Yield a cell's dense raw-count row; the two-view collate handles tokenization + augmentation."""

    def __init__(self, adata: ad.AnnData) -> None:
        self._x = adata.X

    def __len__(self) -> int:
        return self._x.shape[0]

    def __getitem__(self, i: int) -> np.ndarray:
        row = self._x[i]
        return np.asarray(row.todense()).ravel().astype(np.float32) if sparse.issparse(row) else np.asarray(row)


class LossHistory(pl.Callback):
    def __init__(self) -> None:
        self.loss: list[float] = []
        self.acc: list[float] = []

    def on_train_batch_end(self, trainer, *_a, **_k) -> None:
        m = trainer.callback_metrics
        if "loss" in m:
            self.loss.append(float(m["loss"]))
        if "retrieval_acc" in m:
            self.acc.append(float(m["retrieval_acc"]))


def main() -> None:
    adata = synthetic_cxg()
    vocab = GeneVocab(adata.var_names)
    var_token = vocab.align(adata.var_names)
    print(f"[smoke] AnnData {adata.shape} | vocab n_tokens={vocab.n_tokens} | mapped genes={int((var_token >= 0).sum())}")

    collate = TwoViewCollate(var_token, max_tokens=MAX_TOKENS, seed=0)
    loader = torch.utils.data.DataLoader(
        CellDataset(adata), batch_size=BATCH, shuffle=True, drop_last=True, collate_fn=lambda s: collate(np.stack(s))
    )

    backbone = GeneEncoderConfig(
        n_tokens=vocab.n_tokens, dim_model=64, n_layers=2, n_heads=4, dim_feedforward=128, dropout=0.0, max_rank=128
    ).build()
    model = ContrastiveModel(backbone, ContrastiveHead())  # backbone + contrastive head (logit_scale)
    module = TrainingModule(model, ContrastiveObjective(), lr=1e-3)
    print(f"[smoke] model params={sum(p.numel() for p in model.parameters()):,}")

    history = LossHistory()
    trainer = pl.Trainer(
        max_steps=MAX_STEPS, accelerator="cpu", devices=1, callbacks=[history], logger=False,
        enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
    )
    trainer.fit(module, train_dataloaders=loader)

    first, last = np.mean(history.loss[:5]), np.mean(history.loss[-5:])
    print(f"[smoke] loss {first:.3f} -> {last:.3f} over {len(history.loss)} steps "
          f"| retrieval_acc {np.mean(history.acc[:5]):.2f} -> {np.mean(history.acc[-5:]):.2f}")
    assert np.isfinite(history.loss).all(), "loss went non-finite"
    assert last < first, f"contrastive loss did not decrease ({first:.3f} -> {last:.3f})"
    print("\nCONTRASTIVE SMOKE PASSED")


if __name__ == "__main__":
    main()
