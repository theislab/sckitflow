"""Train FlowMatching on Tahoe and report whether the loss decreases + how fast (steps/s).

A real training entry point (not a pytest): fits on the converted Tahoe plates (or a subset) with the
recommended loader config, logs a smoothed train-loss curve + steady-state throughput, and optionally
scores a held-IN quality check (translate control→drug for a few training conditions and compare the
predicted vs. real perturbed population with r_squared / e-distance). Use it to sanity-check that a
config actually learns and to compare speeds across knobs.

Usage
-----
    python scripts/tahoe_train.py --plates '/lustre/.../tahoe100_converted/plate3_*.zarr' \
        --n-train-steps 3000 --batch-size 1024 --chunk-size 16 --device cuda
    python scripts/tahoe_train.py --plates '/lustre/.../plate*.zarr' --amp bf16 --compile
"""

from __future__ import annotations

import argparse
import glob
import statistics as st
import sys
import time

import numpy as np


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plates", required=True, help="comma-separated glob(s) of Tahoe zarr plates")
    p.add_argument("--sample-rep", default="X_pca", help="obsm rep streamed as the state")
    p.add_argument("--device", default="cuda")
    p.add_argument("--objective", choices=["otfm", "genot"], default="otfm")
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--chunk-size", type=int, default=16)
    p.add_argument("--min-runs-per-leaf", type=int, default=None, help="default = chunk_size")
    p.add_argument("--preload-nchunks", type=int, default=None)
    p.add_argument("--control-in-memory", action="store_true", help="materialize controls in RAM (enables larger chunk_size)")
    p.add_argument("--condition-embedding-dim", type=int, default=64)
    p.add_argument("--match-context", default="cell_line",
                    help="comma-separated match_context columns; add 'plate' to keep OT matching within-plate")
    p.add_argument("--hidden-dims", default="1024,1024,1024")
    p.add_argument("--regularization", type=float, default=0.0,
                    help="condition-embedding L2 penalty; default 0.0 (FlowMatching's own default of 1.0 "
                         "collapses drug embeddings -> model degenerates to the identity baseline)")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--n-train-steps", type=int, default=3000)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--split-by", default=None, help="e.g. 'drug' — hold out whole conditions for validation")
    p.add_argument("--valid-freq", type=int, default=2000, help="run the held-out validation pass every N steps")
    p.add_argument("--val-num-steps", type=int, default=50, help="ODE integration steps for validation")
    p.add_argument("--n-val-conditions", type=int, default=None)
    p.add_argument("--metrics", default="r_squared,e-dist")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    import lightning.pytorch as pl
    import torch

    from sc_flow import FlowMatching
    from sc_flow.data import FlowSpec
    from sc_flow.data._encoders import one_hot
    from sc_flow.data.schemas import ConditionDataSchema, StateDataSchema

    paths = sorted({p for pattern in args.plates.split(",") for p in glob.glob(pattern)})
    if not paths:
        print(f"[train] FAIL: no plates matched {args.plates!r}", flush=True)
        return 1
    chunk = args.chunk_size
    min_runs = args.min_runs_per_leaf if args.min_runs_per_leaf is not None else (chunk if chunk > 1 else 0)
    hidden = tuple(int(x) for x in args.hidden_dims.split(",") if x)
    print(f"[train] plates={len(paths)} rep={args.sample_rep} objective={args.objective} batch={args.batch_size} "
          f"chunk={chunk} min_runs={min_runs} match_context={args.match_context} hidden={hidden} "
          f"emb={args.condition_embedding_dim} reg={args.regularization} lr={args.lr} "
          f"amp=off compile=off device={args.device}", flush=True)

    spec = FlowSpec(
        state=StateDataSchema(sample_rep=args.sample_rep),
        condition=ConditionDataSchema(conditions={"drug": ["drug"]}, condition_encoders={"drug": one_hot()}),
        control_key="is_control",
        match_context=args.match_context.split(","),
    )
    model = FlowMatching(
        spec=spec, objective=args.objective, condition_embedding_dim=args.condition_embedding_dim,
        hidden_dims=hidden, regularization=args.regularization, seed=args.seed,
    )

    class LossLogger(pl.Callback):
        """Record per-step train loss + wall time; report a smoothed curve and steady-state steps/s."""

        def __init__(self, log_every: int):
            self.losses: list[float] = []
            self.tstep: list[float] = []
            self._t0 = None
            self._log_every = log_every

        def on_train_batch_start(self, *a):
            self._t0 = time.perf_counter()

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            if self._t0 is not None:
                self.tstep.append(time.perf_counter() - self._t0)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs
            self.losses.append(float(loss))
            n = len(self.losses)
            if n % self._log_every == 0:
                window = self.losses[-self._log_every:]
                steady = self.tstep[20:] if len(self.tstep) > 20 else self.tstep
                sps = 1.0 / st.median(steady) if steady else float("nan")
                print(f"[train]   step {n:5d}  loss(mean last {self._log_every})={np.mean(window):.4f}  "
                      f"steps/s(median)={sps:.1f}", flush=True)

    class ValLogger(pl.Callback):
        """Print the held-out r_squared/e-dist etc. every validation pass (mid-training, not just at the end)."""

        def on_validation_epoch_end(self, trainer, pl_module):
            hist = pl_module.metrics_history
            if not hist or not any(v for v in hist.values()):
                return
            msg = "  ".join(f"{name}={values[-1]:+.4f}" for name, values in hist.items() if values)
            print(f"[train] VAL  step {trainer.global_step:7d}  {msg}", flush=True)

    logger = LossLogger(args.log_every)
    callbacks = [logger]
    split_kwargs: dict = {}
    if args.split_by:
        split_kwargs = dict(
            split_by=args.split_by, valid_freq=args.valid_freq, val_num_steps=args.val_num_steps,
            n_val_conditions=args.n_val_conditions, metrics=tuple(args.metrics.split(",")),
        )
        callbacks.append(ValLogger())
        print(f"[train] validation ON: split_by={args.split_by} valid_freq={args.valid_freq} "
              f"metrics={split_kwargs['metrics']} val_num_steps={args.val_num_steps}", flush=True)

    t0 = time.perf_counter()
    model.fit(
        paths, rep_tables=None, batch_size=args.batch_size, chunk_size=chunk, min_runs_per_leaf=min_runs,
        preload_nchunks=args.preload_nchunks, control_in_memory=args.control_in_memory,
        n_train_steps=args.n_train_steps, device=args.device, lr=args.lr, callbacks=callbacks, **split_kwargs,
    )
    dt = time.perf_counter() - t0

    losses = logger.losses
    if len(losses) < 20:
        print("[train] too few steps to judge", flush=True)
        return 1
    first = float(np.mean(losses[: max(10, len(losses) // 20)]))
    last = float(np.mean(losses[-max(10, len(losses) // 20):]))
    steady = logger.tstep[20:] if len(logger.tstep) > 20 else logger.tstep
    sps = 1.0 / st.median(steady) if steady else float("nan")
    print(f"[train] DONE {args.n_train_steps} steps in {dt:.1f}s | steady {sps:.1f} steps/s | "
          f"loss {first:.4f} -> {last:.4f} ({'DECREASED' if last < first else 'did NOT decrease'})", flush=True)
    if model.metrics_history:
        for name, values in model.metrics_history.items():
            print(f"[train] VAL {name} trajectory ({len(values)} passes): "
                  f"{values[0]:+.4f} -> {values[-1]:+.4f}", flush=True)
    return 0 if last < first else 2


if __name__ == "__main__":
    sys.exit(main())
