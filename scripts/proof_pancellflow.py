"""PROOF: pan-cell flow — one foundation encoder gives flow a shared latent across DIFFERENT gene panels.

Two synthetic datasets with **disjoint gene panels** (A: genes 0..55, B: 56..111) express the same
control/perturbed biology through different genes. The pan-cell model runs both through ONE
sc_flow.concept.GeneEncoder (state-encoder slot) into one latent, and ONE rectified-flow velocity transports
control→perturbed there. We show: (1) it trains across both panels and the loss drops; (2) the learned flow
moves control cells toward the perturbed manifold for BOTH datasets (cross-dataset transport in the shared
space); (3) it works with the encoder fine-tuned AND frozen; (4) it's dispatched generically via
sc_flow.families.build_family("pancell", recipe).

Run (CPU):  python scripts/proof_pancellflow.py
"""

from __future__ import annotations

import anndata as ad
import lightning.pytorch as pl
import numpy as np
import torch
from scipy import sparse

from sc_flow.families import available_families, build_family
from sc_flow.pancell._data import tokenize_batch

PANEL, PROGRAMS, MARKERS = 56, 8, 7  # genes per panel, programs, markers/program/panel
N_GENES = 2 * PANEL
CTRL_PROGS, PERT_PROGS = [0, 1, 2, 3], [4, 5, 6, 7]


def make_two_panel(seed: int = 0) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    a_mark = {k: np.arange(k * MARKERS, (k + 1) * MARKERS) for k in range(PROGRAMS)}
    b_mark = {k: PANEL + np.arange(k * MARKERS, (k + 1) * MARKERS) for k in range(PROGRAMS)}

    def cell(panel: str, progs: list[int]) -> np.ndarray:
        x = np.zeros(N_GENES, np.float32)
        mark = a_mark if panel == "A" else b_mark
        genes = mark[int(rng.choice(progs))]
        x[genes] = rng.poisson(6.0, size=len(genes)).astype(np.float32)
        bg = np.arange(0, PANEL) if panel == "A" else np.arange(PANEL, N_GENES)
        x[rng.choice(bg, 3, replace=False)] += rng.poisson(1.0, 3).astype(np.float32)
        return x

    rows, is_ctrl, dset = [], [], []
    for panel in ("A", "B"):
        for _ in range(150):
            rows.append(cell(panel, CTRL_PROGS)); is_ctrl.append(True); dset.append(panel)
        for _ in range(150):
            rows.append(cell(panel, PERT_PROGS)); is_ctrl.append(False); dset.append(panel)
    adata = ad.AnnData(X=sparse.csr_matrix(np.stack(rows)))
    adata.var_names = [f"ENSG{i:011d}" for i in range(N_GENES)]
    adata.obs["is_control"] = is_ctrl
    adata.obs["dataset"] = dset
    adata.obs_names = [str(i) for i in range(len(rows))]
    return adata


class LossHistory(pl.Callback):
    def __init__(self):
        self.loss = []

    def on_train_batch_end(self, trainer, *_a, **_k):
        if "loss" in trainer.callback_metrics:
            self.loss.append(float(trainer.callback_metrics["loss"]))


def dense(a) -> np.ndarray:
    return np.asarray(a.X.todense()) if sparse.issparse(a.X) else np.asarray(a.X)


@torch.no_grad()
def transport(model, rows, var_token, valid, max_tokens, pert_mean, steps=20):
    """Euler-integrate the learned velocity from control latents; distance to the perturbed mean before/after."""
    tok, mask = tokenize_batch(rows, var_token, valid, max_tokens)
    z0 = model.encode(tok, mask)
    z, dt = z0.clone(), 1.0 / steps
    for i in range(steps):
        z = z + dt * model.velocity(z, torch.full((z.shape[0], 1), i * dt))
    return (z0 - pert_mean).norm(dim=-1).mean().item(), (z - pert_mean).norm(dim=-1).mean().item()


def run(adata, *, freeze: bool, steps: int = 500, seed: int = 0):
    recipe = {
        "data": {"adata": adata, "control_key": "is_control"},
        "state_encoder": {"dim_model": 64, "n_layers": 2, "n_heads": 4, "dim_feedforward": 128,
                          "max_rank": 65, "dropout": 0.0, "freeze": freeze},
        "velocity": {"hidden": 128, "n_layers": 3},
        "sampler": {"batch_size": 128, "max_tokens": 64, "steps_per_epoch": steps},
        "trainer": {"lr": 1e-3},
        "seed": seed,
    }
    builder = build_family("pancell", recipe)  # generic dispatch through the family registry
    hist = LossHistory()
    trainer = pl.Trainer(max_steps=steps, accelerator="cpu", devices=1, callbacks=[hist], logger=False,
                         enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False)
    trainer.fit(builder.module, datamodule=builder.datamodule)

    model = builder.model
    vocab = builder.datamodule  # reuse its aligned var_token
    var_token, valid = vocab._var_token, vocab._valid
    perturbed = dense(adata[~adata.obs.is_control.to_numpy()])
    with torch.no_grad():
        pt, pm = tokenize_batch(perturbed, var_token, valid, 64)
        pert_mean = model.encode(pt, pm).mean(0, keepdim=True)
    a_ctrl = dense(adata[(adata.obs.is_control) & (adata.obs.dataset == "A")])
    b_ctrl = dense(adata[(adata.obs.is_control) & (adata.obs.dataset == "B")])
    a_before, a_after = transport(model, a_ctrl, var_token, valid, 64, pert_mean)
    b_before, b_after = transport(model, b_ctrl, var_token, valid, 64, pert_mean)
    return hist.loss, (a_before, a_after), (b_before, b_after)


def main():
    torch.manual_seed(0)
    adata = make_two_panel()
    a_only = adata[adata.obs.dataset == "A"]
    b_only = adata[adata.obs.dataset == "B"]
    a_genes = np.flatnonzero(dense(a_only).sum(0) > 0)
    b_genes = np.flatnonzero(dense(b_only).sum(0) > 0)
    print(f"[data] {adata.shape} | 2 datasets, DISJOINT panels: A uses genes {a_genes.min()}..{a_genes.max()}, "
          f"B uses {b_genes.min()}..{b_genes.max()} (overlap={len(np.intersect1d(a_genes, b_genes))})")
    print(f"[families] discoverable: {available_families()}\n")

    ok = True
    for freeze in (False, True):
        mode = "FROZEN encoder (velocity-only; a real run loads a pretrained bundle here)" if freeze \
            else "JOINT fine-tune (encoder + velocity)"
        loss, (a0, a1), (b0, b1) = run(adata, freeze=freeze)
        first, last = float(np.mean(loss[:15])), float(np.mean(loss[-15:]))
        print(f"=== {mode} ===")
        print(f"  fm loss {first:.3f} -> {last:.3f} over {len(loss)} steps")
        print(f"  dataset A control → perturbed-manifold distance {a0:.3f} -> {a1:.3f}   (transport in shared latent)")
        print(f"  dataset B control → perturbed-manifold distance {b0:.3f} -> {b1:.3f}")
        drop = last < first
        moves = a1 < a0 and b1 < b0
        print(f"  loss decreased: {drop} | cross-dataset transport (A & B move toward perturbed): {moves}\n")
        ok = ok and drop and moves

    print("PAN-CELL FLOW PROOF PASSED" if ok else "PROOF FAILED")
    assert ok


if __name__ == "__main__":
    main()
