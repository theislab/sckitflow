"""End-to-end contrastive pretraining of ``sc_flow.concept.FoundationModel`` on real CellxGene h5ad(s).

NOT a smoke: reads real raw-count cells (Ensembl ``var``, sparse CSR) and trains the full new stack —
Component-configured GeneEncoder backbone + ContrastiveHead + CLIP objective, assembled by the
``FoundationModel`` builder, trained with plain ``lightning.Trainer.fit`` on GPU (or CPU).

  # cluster GPU (one real CxG file):
  python scripts/train_concept_cxg.py --h5ad /lustre/.../file.h5ad --device gpu --precision bf16-mixed \
      --steps 3000 --batch-size 256 --max-tokens 512 --out $APP_RUNS_DIR/concept_cxg

  # local CPU check on a synthetic CxG-shaped AnnData (no data needed):
  python scripts/train_concept_cxg.py --synthetic --device cpu --steps 100

Full multi-species 2.5 TB streaming is the documented scale-up; this trains one (or a few concatenated)
in-memory dataset(s) — a real run that exercises the whole architecture.
"""

from __future__ import annotations

import argparse
import json
import time

import anndata as ad
import lightning.pytorch as pl
import numpy as np


def load_h5ads(paths: list[str]) -> ad.AnnData:
    parts = []
    for p in paths:
        a = ad.read_h5ad(p)
        print(f"[data] {p}: {a.shape} X={type(a.X).__name__}", flush=True)
        parts.append(a)
    if len(parts) == 1:
        return parts[0]
    merged = ad.concat(parts, join="outer", fill_value=0)  # union of genes across shards
    print(f"[data] concatenated -> {merged.shape} (outer gene join)", flush=True)
    return merged


def synthetic(n_cells: int = 512, n_genes: int = 300, n_programs: int = 24, seed: int = 0) -> ad.AnnData:
    from scipy import sparse

    rng = np.random.default_rng(seed)
    program_genes = [rng.choice(n_genes, size=25, replace=False) for _ in range(n_programs)]
    cell_program = rng.integers(0, n_programs, size=n_cells)
    x = np.zeros((n_cells, n_genes), dtype=np.float32)
    for i in range(n_cells):
        rate = np.full(n_genes, 0.2)
        rate[program_genes[cell_program[i]]] += rng.uniform(2.0, 5.0)
        x[i] = rng.poisson(rate * rng.uniform(0.5, 1.5)).astype(np.float32)
    a = ad.AnnData(X=sparse.csr_matrix(x))
    a.var_names = [f"ENSG{i:011d}" for i in range(n_genes)]
    a.obs_names = [str(i) for i in range(n_cells)]
    return a


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
    ap = argparse.ArgumentParser(description="Contrastive FoundationModel pretraining on CellxGene h5ad(s).")
    ap.add_argument("--h5ad", nargs="+", help="one or more CxG h5ad files (raw counts, Ensembl var)")
    ap.add_argument("--synthetic", action="store_true", help="use a synthetic CxG-shaped AnnData (local check)")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--dim-model", type=int, default=256)
    ap.add_argument("--n-layers", type=int, default=6)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cpu", help="cpu | gpu | mps")
    ap.add_argument("--precision", default="32-true", help="e.g. bf16-mixed on A100")
    ap.add_argument("--species", default="unknown")
    ap.add_argument("--out", default=None, help="dir to save the model bundle")
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()

    if not args.synthetic and not args.h5ad:
        ap.error("pass --h5ad <files...> or --synthetic")

    from sc_flow.concept import FoundationModel

    t0 = time.time()
    adata = synthetic() if args.synthetic else load_h5ads(args.h5ad)
    print(f"[data] final AnnData {adata.shape} loaded in {time.time()-t0:.1f}s", flush=True)

    recipe = {
        "data": {"adata": adata, "species": args.species},
        "backbone": {
            "dim_model": args.dim_model, "n_layers": args.n_layers, "n_heads": args.n_heads,
            "dim_feedforward": args.dim_model * 4, "dropout": 0.1, "max_rank": args.max_tokens + 1,
        },
        "objective": {"logit_scale_init": 3.0, "max_logit_scale": 100.0},
        "sampler": {"batch_size": args.batch_size, "max_tokens": args.max_tokens},
        "trainer": {"lr": args.lr},
        "task": "contrastive",
        "seed": 0,
    }
    fm = FoundationModel(recipe)
    n_params = sum(p.numel() for p in fm.module.model.parameters())
    print(f"[model] vocab n_tokens={fm.vocab.n_tokens} | params={n_params:,} | device={args.device}", flush=True)

    logger = False
    if args.wandb:
        try:
            from lightning.pytorch.loggers import WandbLogger

            logger = WandbLogger(project="cf-train-concept")
        except Exception as e:  # noqa: BLE001
            print(f"[warn] wandb unavailable ({e}); continuing tracker-free", flush=True)

    history = LossHistory()
    trainer = pl.Trainer(
        max_steps=args.steps, accelerator=args.device, devices=1, precision=args.precision,
        callbacks=[history], logger=logger, enable_checkpointing=False, enable_progress_bar=False,
        enable_model_summary=False, log_every_n_steps=25,
    )
    trainer.fit(fm.module, datamodule=fm.datamodule)

    if history.loss:
        f, l = np.mean(history.loss[:10]), np.mean(history.loss[-10:])
        fa, la = np.mean(history.acc[:10] or [0]), np.mean(history.acc[-10:] or [0])
        print(f"[done] loss {f:.3f} -> {l:.3f} | retrieval_acc {fa:.2f} -> {la:.2f} | "
              f"{len(history.loss)} steps in {time.time()-t0:.0f}s", flush=True)
    if args.out:
        fm.save(args.out)
        (__import__("pathlib").Path(args.out) / "history.json").write_text(
            json.dumps({"loss": history.loss, "acc": history.acc}))
        print(f"[done] saved bundle -> {args.out}", flush=True)
    print("TRAIN_CONCEPT_CXG DONE", flush=True)


if __name__ == "__main__":
    main()
