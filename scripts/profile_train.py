"""Profile a FlowMatching training step to find + rank the hotspots (data / coupling / fwd+bwd).

Two measurements:
  1. **Throughput** — median wall-time per optimizer step over ``--n-steps`` (after warmup), i.e. steps/s.
  2. **Breakdown** — a torch profiler trace (CPU+CUDA) over ``--profile-steps`` steps, with named regions
     (``data``, ``objective`` / coupling, ``fwd+bwd``) via ``record_function``, printed as a self-CUDA-time
     table + the region totals. This is what tells you whether the per-step cost is the torch↔jax OT
     coupling, the data loader (zarr streaming), or the network fwd/bwd.

Works on synthetic data by default (so it runs anywhere), or on a real dataset via ``--data <path>`` (a
zarr adata / list of zarr plate paths) plus the Tahoe schema flags. Run on a GPU node with
``CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false``.

Usage
-----
    python scripts/profile_train.py --device cuda --n-steps 60 --profile-steps 8
    python scripts/profile_train.py --device cuda --data /lustre/.../tahoe_converted \
        --sample-rep X_pca --drug-col drug --cell-col cell_line --control-key is_control \
        --objective otfm --batch-size 1024
"""

from __future__ import annotations

import argparse
import statistics
import time
from contextlib import nullcontext

import numpy as np


def _synth_adata(*, n_per: int, d: int, seed: int):
    import anndata as ad
    import pandas as pd

    rng = np.random.default_rng(seed)
    cts = {"cl_a": 0.0, "cl_b": 2.0}
    drugs = {"drug_a": +5.0, "drug_b": -5.0}
    xs, ct, dr, ctrl = [], [], [], []
    for c, mu in cts.items():
        xs.append(rng.normal(mu, 0.3, (n_per, d)).astype(np.float32)); ct += [c] * n_per; dr += ["control"] * n_per; ctrl += [True] * n_per
        for name, s in drugs.items():
            xs.append((rng.normal(mu, 0.3, (n_per, d)) + s).astype(np.float32)); ct += [c] * n_per; dr += [name] * n_per; ctrl += [False] * n_per
    obs = pd.DataFrame({"cell_type": ct, "drug1": dr, "control": ctrl})
    for c in ("cell_type", "drug1"):
        obs[c] = obs[c].astype("category")
    adata = ad.AnnData(X=np.concatenate(xs, 0), obs=obs)
    adata.uns["drug"] = {k: rng.standard_normal((1, 4)).astype(np.float32) for k in obs["drug1"].cat.categories}
    return adata


def _build(args):
    from sc_flow import FlowMatching
    from sc_flow.core.data import FlowSpec
    from sc_flow.core.data._encoders import lookup, one_hot
    from sc_flow.core.data.schemas import ConditionDataSchema, StateDataSchema

    if args.data:  # real dataset (Tahoe): one-hot the drug covariate, match on cell line
        spec = FlowSpec(
            state=StateDataSchema(sample_rep=args.sample_rep),
            condition=ConditionDataSchema(
                conditions={args.drug_col: [args.drug_col]}, condition_encoders={args.drug_col: one_hot()}
            ),
            control_key=args.control_key,
            match_context=[args.cell_col] if args.cell_col else [],
        )
        data, rep_tables = args.data, None
    else:  # synthetic
        spec = FlowSpec(
            state=StateDataSchema(sample_rep="X"),
            condition=ConditionDataSchema(conditions={"drug": ["drug1"]}, condition_encoders={"drug": lookup("drug")}),
            control_key="control",
            match_context=["cell_type"],
        )
        adata = _synth_adata(n_per=args.n_per, d=args.d, seed=0)
        data, rep_tables = adata, adata.uns
    model = FlowMatching(
        spec=spec, objective=args.objective, condition_embedding_dim=args.emb_dim, hidden_dims=tuple(args.hidden),
    )
    return model, data, rep_tables


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="cpu")
    p.add_argument("--objective", choices=["otfm", "genot"], default="otfm")
    p.add_argument("--data", default=None, help="real dataset path (zarr adata / list); omit for synthetic")
    p.add_argument("--sample-rep", default="X_pca")
    p.add_argument("--drug-col", default="drug")
    p.add_argument("--cell-col", default="cell_line")
    p.add_argument("--control-key", default="is_control")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--chunk-size", type=int, default=1, help="contiguous cells/chunk (>1 needs grouped data)")
    p.add_argument("--emb-dim", type=int, default=64)
    p.add_argument("--hidden", type=int, nargs="+", default=[512, 512])
    p.add_argument("--n-steps", type=int, default=60, help="steps for the throughput measurement")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--profile-steps", type=int, default=8, help="steps for the torch-profiler breakdown")
    p.add_argument("--n-per", type=int, default=2048, help="synthetic cells per (ct,cond)")
    p.add_argument("--d", type=int, default=128, help="synthetic state dim")
    args = p.parse_args()

    import torch
    from binded import Loader, SamplerConfig

    from sc_flow.core.training._harness import SCFlowLightningModule
    from sc_flow.core.training._objective import build_objective

    print(f"[profile] device={args.device} objective={args.objective} batch={args.batch_size} "
          f"data={'synthetic' if not args.data else args.data}", flush=True)

    model, data, rep_tables = _build(args)
    device = torch.device(args.device)

    # build the pieces directly so we can drive the step loop ourselves (bypass Lightning) for clean timing
    compiled = model.spec.compile(data, rep_tables=rep_tables, seed=0)
    model._dims = compiled.dims
    model._condition_fn = compiled.condition_fn
    vf = model._build_vf(compiled.dims).to(device)
    path = model._build_probability_path()
    objective = build_objective(
        model.objective_name, path, condition_mode=model.condition_mode, regularization=model.regularization,
        coupling_locs=compiled.coupling, match_method=model.match_method, match_kwargs=model.match_kwargs, seed=0,
    )
    harness = SCFlowLightningModule(vf, objective, lr=1e-4)
    opt = harness.configure_optimizers()
    _preload = max(1, args.batch_size) if args.chunk_size <= 1 else max(args.chunk_size, 4 * (args.batch_size // args.chunk_size))
    cfg = SamplerConfig(batch_size=args.batch_size, chunk_size=args.chunk_size, preload_nchunks=_preload, to=None)
    loader = Loader(compiled.scheme, cfg, compiled.condition_fn)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    it = iter(loader)

    def one_step(regions: bool):
        rf = torch.profiler.record_function if regions else (lambda _n: nullcontext())
        with rf("data"):
            batch = next(it)
        opt.zero_grad(set_to_none=True)
        with rf("objective"):  # includes the torch<->jax OT coupling + fwd
            loss, _ = objective.compute_loss(vf, batch)
        with rf("fwd_bwd"):
            loss.backward()
            opt.step()
        return float(loss.detach())

    # --- warmup (also triggers jax jit of the coupling) ---
    for _ in range(args.warmup):
        one_step(False)
    sync()

    # --- throughput ---
    times = []
    for _ in range(args.n_steps):
        t0 = time.perf_counter()
        one_step(False)
        sync()
        times.append(time.perf_counter() - t0)
    med = statistics.median(times)
    print(f"[profile] steps/s={1.0 / med:.2f}  median_step={med * 1e3:.1f}ms  "
          f"p10={np.percentile(times, 10) * 1e3:.1f}ms p90={np.percentile(times, 90) * 1e3:.1f}ms", flush=True)

    # --- profiler breakdown ---
    acts = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        acts.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=acts, record_shapes=False, with_stack=False) as prof:
        for _ in range(args.profile_steps):
            one_step(True)
        sync()
    sort_key = "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
    print("\n[profile] top ops by " + sort_key + ":", flush=True)
    print(prof.key_averages().table(sort_by=sort_key, row_limit=18))
    # region totals (data / objective / fwd_bwd)
    print("[profile] region totals (from record_function):", flush=True)
    for evt in prof.key_averages():
        if evt.key in ("data", "objective", "fwd_bwd"):
            cuda = getattr(evt, "cuda_time_total", 0) or getattr(evt, "device_time_total", 0)
            print(f"    {evt.key:10s} cpu_total={evt.cpu_time_total / 1e3:8.1f}ms cuda_total={cuda / 1e3:8.1f}ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
