"""Tahoe smoke test: train + validate FlowMatching on the Tahoe data contract.

Mirrors the real Tahoe-100M setup (cf-train ``configs/experiments/tahoe.yaml``): an ``obsm/X_pca``
state rep, ``drug`` perturbation covariate, ``cell_line`` split/match context, and a boolean
``is_control`` (drug == ``DMSO_TF``). Two modes:

* **sim-tahoe** (default) — a small synthetic plate with the same schema, fast enough to gate a
  laptop CPU or a GPU node. Each drug carries a random feature vector (as real Tahoe carries drug
  embeddings) and applies a shift ``= W · feature`` to control cells, so the model can *generalize to
  held-out drugs* — the held-out-split validation R²/E-distance are a real learning signal, not noise.
* **real** (``--plates '<glob>'``) — streams the converted Tahoe zarr plates.

Exits non-zero if the model fails to learn (sim mode), so it doubles as a "did this machine set up the
numeric stack correctly" gate on CPU and GPU.

Usage
-----
    python scripts/tahoe_smoke.py --device cpu
    python scripts/tahoe_smoke.py --device cuda --objective otfm --n-train-steps 800
    python scripts/tahoe_smoke.py --device cuda --plates \
        '/lustre/groups/ml01/datasets/selman.ozleyen/tahoe100_converted/plate3_*.zarr' --n-train-steps 2000
"""

from __future__ import annotations

import argparse
import glob
import sys
import time

import anndata as ad
import numpy as np
import pandas as pd

from sc_flow import FlowMatching
from sc_flow.data import FlowSpec
from sc_flow.data._encoders import lookup, one_hot
from sc_flow.data.schemas import ConditionDataSchema, StateDataSchema

CONTROL_DRUG = "DMSO_TF"  # the Tahoe control drug (is_control := drug == DMSO_TF)


def make_sim_tahoe(
    *, n_cell_lines: int, n_drugs: int, n_per: int, pca_dim: int, feat_dim: int, delta: float, seed: int
) -> tuple[ad.AnnData, np.ndarray, dict]:
    """Synthetic Tahoe-shaped plate with a learnable, held-out-generalizable per-drug shift.

    Returns ``(adata, W, drug_features)``: ``adata`` has ``obsm['X_pca']`` + obs
    ``drug``/``cell_line``/``is_control`` and ``uns['drug']`` feature table; ``W`` maps a drug feature
    to its PCA-space shift (``shift_d = delta * W @ feature_d``), applied to control cells.
    """
    rng = np.random.default_rng(seed)
    cell_lines = [f"cl_{i}" for i in range(n_cell_lines)]
    drugs = [f"drug_{i}" for i in range(n_drugs)]
    cl_mean = {cl: rng.normal(0.0, 1.0, pca_dim).astype(np.float32) for cl in cell_lines}
    drug_features = {d: rng.standard_normal(feat_dim).astype(np.float32) for d in drugs}
    drug_features[CONTROL_DRUG] = np.zeros(feat_dim, np.float32)  # control has no perturbation feature
    W = rng.standard_normal((pca_dim, feat_dim)).astype(np.float32) / np.sqrt(feat_dim)

    X, cl_col, dr_col, ctrl_col = [], [], [], []
    for cl in cell_lines:
        base = cl_mean[cl]
        # control cells for this line
        X.append((rng.normal(0.0, 0.3, (n_per, pca_dim)).astype(np.float32) + base))
        cl_col += [cl] * n_per; dr_col += [CONTROL_DRUG] * n_per; ctrl_col += [True] * n_per
        for d in drugs:
            shift = (delta * (W @ drug_features[d])).astype(np.float32)
            X.append((rng.normal(0.0, 0.3, (n_per, pca_dim)).astype(np.float32) + base + shift))
            cl_col += [cl] * n_per; dr_col += [d] * n_per; ctrl_col += [False] * n_per

    obs = pd.DataFrame({"cell_line": cl_col, "drug": dr_col, "is_control": ctrl_col})
    for c in ("cell_line", "drug"):
        obs[c] = obs[c].astype("category")
    adata = ad.AnnData(X=np.zeros((obs.shape[0], 1), np.float32), obs=obs)
    adata.obsm["X_pca"] = np.concatenate(X, axis=0)
    # uns['drug'] feature table for the lookup encoder (each value shape (1, feat_dim)).
    adata.uns["drug"] = {d: f[None, :] for d, f in drug_features.items()}
    return adata, W, drug_features


def build_spec(*, use_features: bool) -> FlowSpec:
    """Tahoe FlowSpec: X_pca state, drug condition, is_control key, cell_line match context.

    ``use_features`` picks the drug encoder — ``lookup`` (feature vectors, generalizes to held-out
    drugs; used by sim-tahoe) or ``one_hot`` (identity; the simplest real-plate default).
    """
    enc = lookup("drug") if use_features else one_hot()
    return FlowSpec(
        state=StateDataSchema(sample_rep="X_pca"),
        condition=ConditionDataSchema(conditions={"drug": ["drug"]}, condition_encoders={"drug": enc}),
        control_key="is_control",
        match_context=["cell_line"],
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--objective", choices=["otfm", "genot"], default="otfm")
    p.add_argument("--device", default="cpu")
    p.add_argument("--plates", default=None, help="glob of real Tahoe zarr plates; omit for sim-tahoe")
    p.add_argument("--n-cell-lines", type=int, default=3)
    p.add_argument("--n-drugs", type=int, default=8)
    p.add_argument("--n-per", type=int, default=256, help="cells per (cell_line, drug) leaf")
    p.add_argument("--pca-dim", type=int, default=32)
    p.add_argument("--feat-dim", type=int, default=6)
    p.add_argument("--delta", type=float, default=4.0)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--chunk-size", type=int, default=1,
                   help="contiguous cells/read; >1 needs data grouped into per-condition runs (must divide batch)")
    p.add_argument("--n-train-steps", type=int, default=600)
    p.add_argument("--valid-freq", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    print(f"[tahoe_smoke] mode={'real' if args.plates else 'sim'} device={args.device!r} "
          f"objective={args.objective!r}", flush=True)

    if args.plates:
        paths = sorted(glob.glob(args.plates))
        if not paths:
            print(f"[tahoe_smoke] FAIL: no plates matched {args.plates!r}", flush=True)
            return 1
        print(f"[tahoe_smoke] {len(paths)} plate(s): {[p.split('/')[-1] for p in paths]}", flush=True)
        data, rep_tables, spec = paths, None, build_spec(use_features=False)
        controls = None
    else:
        adata, _W, _feats = make_sim_tahoe(
            n_cell_lines=args.n_cell_lines, n_drugs=args.n_drugs, n_per=args.n_per,
            pca_dim=args.pca_dim, feat_dim=args.feat_dim, delta=args.delta, seed=args.seed,
        )
        print(f"[tahoe_smoke] sim adata: {adata.shape} X_pca={adata.obsm['X_pca'].shape} "
              f"cell_lines={args.n_cell_lines} drugs={args.n_drugs}", flush=True)
        data, rep_tables, spec = adata, adata.uns, build_spec(use_features=True)
        controls = adata[adata.obs["is_control"].to_numpy()]

    model = FlowMatching(
        spec=spec, objective=args.objective, condition_embedding_dim=32, hidden_dims=(128, 128), seed=args.seed
    )

    t0 = time.perf_counter()
    model.fit(
        data, rep_tables=rep_tables, batch_size=args.batch_size, chunk_size=args.chunk_size,
        n_train_steps=args.n_train_steps, valid_freq=args.valid_freq, device=args.device, lr=args.lr,
        split_by=["drug"], split_ratios={"train": 0.7, "val": 0.3}, val_num_steps=20,
    )
    dt = time.perf_counter() - t0
    steps_s = args.n_train_steps / dt
    print(f"[tahoe_smoke] fit {args.n_train_steps} steps in {dt:.1f}s ({steps_s:.1f} steps/s)", flush=True)
    for name, hist in model.metrics_history.items():
        if hist:
            print(f"[tahoe_smoke]   {name}: {[f'{v:.3f}' for v in hist]}", flush=True)

    if args.plates:
        print("[tahoe_smoke] real-plate run complete (no learning assertion).", flush=True)
        return 0

    # sim gate: on a held-IN condition, the model must move control cells a real distance in PCA space
    # (control cells sit at the cell-line mean; a learned drug applies shift = delta * W @ feature).
    x_ctrl = controls[controls.obs["cell_line"] == "cl_0"].obsm["X_pca"]
    d0 = "drug_0"
    pred = model.predict(x_ctrl, ("cl_0", d0), num_steps=20, seed=0, device=args.device)
    moved_norm = float(np.linalg.norm((pred - x_ctrl).mean(axis=0)))
    print(f"[tahoe_smoke] held-in {d0}: mean shift norm |Δ|={moved_norm:.3f} (delta={args.delta})", flush=True)
    ok = moved_norm > 0.3 * args.delta
    print(f"[tahoe_smoke] {'PASS' if ok else 'FAIL'}: model {'learned' if ok else 'did NOT learn'} "
          f"a condition-dependent shift.", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
