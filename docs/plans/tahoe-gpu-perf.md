# Task: optimize the torch-native FM pipeline to train Tahoe on GPU (perf + cellflow superset)

**Branch:** `perf/tahoe-gpu` (off `feat/refactors`) · **Autonomous run.** Merge to `feat/refactors`/cf-train only
if results are good. Independent agents review the code (esp. tests + lazy deps) with no context.

## Where the project stands (handoff summary)

`sc_flow.FlowMatching` (`src/sc_flow/_model.py`) is a **torch-native** conditional flow-matching model:
VF/encoder/probability-path/loss + Lightning training are torch; the **only** JAX is the minibatch OT
coupling, and it now runs **on-device** — `backends/jax/coupling/_device.couple_device` does torch reps →
JAX via zero-copy DLPack (same device) → ott sinkhorn / quadratic-fused GW + `sample_joint` → torch indices,
so reps + plan stay on `cuda:0`. Objectives: `"otfm"` / `"genot"` (GENOT-L + GENOT-Q), selected by
`FlowMatching(objective=…)`; stochastic condition encoder (`condition_mode="stochastic"`, VAE-KL);
bit-reproducible from one `seed` (CPU). `FlowMatching.save()/load()` persists weights + cloudpickled
spec/dims/condition_fn. Data is the binded layer (`FlowSpec` → `compile_obs` → binded `Loader` streaming from
zarr/h5ad via annbatch). The cellflow/DLPack bridge is deleted; no `import cellflow` in `src`.
Prior plans: `docs/plans/cellflow-vendor-fm.md` (OTFM), `docs/plans/genot-torch-native.md` (GENOT + GPU build).

**Verified on GPU (A100/H100):** otfm + genot fit+predict+learn a conditional shift on `cuda:0`. **Known
hotspot:** on a toy model GPU is ~0.4–0.65 s/step — per-step torch↔jax coupling handoff + jax dispatch
dominate trivial compute. That is the #1 thing to profile/optimize on the real Tahoe model.

## Objectives (this run), priority order

1. **Environment (uv, CUDA, on a GPU node).** Both torch and jax must be CUDA builds matched to the node
   driver: `UV_TORCH_BACKEND=auto` (env var) for torch, `jax[cuda12]` (the `cuda` extra) for jax. `annbatch`
   must stream **zarr** (verify it pulls zarr; add if missing). Run with `XLA_PYTHON_CLIENT_PREALLOCATE=false`.
2. **Lazy / optional deps.** Make heavy backends optional like binded/annbatch already are: importing
   `sc_flow` (and `sc_flow.data`) must not require torch/jax/lightning; each is imported lazily at use with a
   clear "install `sc-flow-tools[…]`" error. Verify `import sc_flow` in a bare env.
3. **Tahoe configs + smoketests.** Wire the real Tahoe paths (from cf-train `configs/experiments/tahoe.yaml`)
   into runnable configs. `scripts/smoke_train.py` already does synthetic; add a **Tahoe** smoke (small
   sim-tahoe plate, GPU) and use Tahoe data as much as possible. Nice smoketests, CPU + GPU.
4. **Profile + optimize speed (the core).** Profile a real Tahoe GPU step: mark hotspots (per-step
   torch↔jax coupling, sinkhorn, DLPack syncs, data loading / zarr streaming, host↔device). Optimize:
   candidate ideas — amortize/async the coupling, avoid per-step recompiles, keep the loader off the critical
   path (num_workers/prefetch), larger batch, `torch.compile` the VF, fp16/bf16, reduce `.item()`/syncs.
   **Measure** steps/s + GPU util (`nvidia-smi dmon`) before/after; log the wins.
5. **Superset cellflow (validations + metrics), math-correct.** Add the deferred validation loop +
   `r_squared` (per-condition predicted-vs-target-mean R²) + `e_distance` (existing `EnergyDistance`), a
   held-out split, and match cellflow's formulations. Scientific/math correctness vs cellflow is a
   requirement — cross-check the OT coupling, the FM/GENOT losses, and the metrics against cellflow's.
6. **Reviews.** Spawn independent no-context agents to review new code and especially **tests** and the
   **lazy-dep** boundaries; fix what they find.

## Environment / access notes

- Cluster: `ssh hpc-submit01.scidom.de '<cmd>'` (banner on stderr, ignore). Never compute on login node.
- GPU session: lab **dropbear** — `sbatch -p gpu_p -q gpu_normal --gres=gpu:1 … --export=ALL,SESSION=scperf
  ~/submit_dropbear.sbatch`, then `ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
  node-scperf '<cmd>'`. The dropbear shell does NOT inherit SLURM's GPU env → prefix `CUDA_VISIBLE_DEVICES=0`.
  Current session node: **supergpu14** (H100, NVLink). Release the GPU when done (no idle allocation).
- Repo staged at `/lustre/groups/ml01/workspace/selman.ozleyen/projects/sc-flow-tools` (rsync working tree,
  exclude `.venv`/`.git`); sibling `../binded` staged too. `uv` at `~/.local/bin/uv`. `annbatch` is pinned by
  git in binded's deps.
- GPU env build: `cd repo && export PATH=~/.local/bin:$PATH UV_LINK_MODE=copy &&
  uv pip install --python .venv/bin/python --torch-backend=auto --reinstall-package torch torch &&
  uv pip install --python .venv/bin/python "jax[cuda12]"`.

## Tahoe data (filled in by investigation)

- **Converted plates (real training data):** `/lustre/groups/ml01/datasets/selman.ozleyen/tahoe100_converted/plate*.zarr`
  (14 plates + `controls.zarr`). Each has `obsm/X_pca` (1024-d, chunks `(64,1024)`, f32) + obs `drug` (380) /
  `cell_line` (50) / `is_control` (= `drug == "DMSO_TF"`). plate3 = 4.7M cells. `annbatch` streams these zarrs
  (verified by the profiler + loader bench).
- **Contract (cf-train `configs/experiments/tahoe.yaml`):** `sample_rep=X_pca`, `perturbation_covariates={drug:[drug]}`,
  `split_covariates=[cell_line]`, `split_by=[drug]`, `control_key=is_control`, `batch_size=1024`, `chunk_size=32`.
  Ported to sc-flow in `configs/tahoe.yaml` (consumed by `scripts/tahoe_smoke.py --config`).
- **sim-tahoe (smoke):** `scripts/tahoe_smoke.py` builds a small grouped plate with the same schema + drug-feature
  shifts (held-out drugs predictable), gated by a conditional-learning check (per-drug `cos(pred,true)`).

## Results (H100 supergpu14, batch=1024, plate3)

**#1 hotspot — the per-step OT coupling ran ott Sinkhorn UNJITTED (eager jax, op-by-op).** Decomposition
(data-load excluded):

| region                       | before (eager) | after (`jax.jit`) | speedup |
|------------------------------|---------------:|------------------:|--------:|
| coupling only                |        404 ms  |            2.5 ms | **162×** |
| full step (couple + fwd/bwd) |        409 ms  |            7.3 ms | **56×**  |

Sinkhorn converges in 10 iters; jitting compiles solve+`sample_joint` into one XLA program (memoized per static
config). ~2.4 → ~140 steps/s of compute. Fix: `backends/jax/coupling/_device.py`. cellflow jits its `match_fn`;
we didn't.

**#2 bottleneck — data loading.** With coupling fixed, the `next(it)`-inclusive step was 2059 ms/step
(0.49 steps/s): the loader used `chunk_size=1` → ~1024 scattered single-row zarr reads/batch (~1985 ms/batch on
Lustre). Fix: `fit()`/`profile_train` now expose `chunk_size`/`preload_nchunks`; on grouped data `chunk_size=32`
reads contiguous runs (cf-train measured ~7.8 → 640 batch/s). The converted single plate isn't grouped into
≥32-cell runs, so it needs the grouped store — sim-tahoe is grouped by construction and validates the fast path.

**Also fixed:** the GPU validation loop crashed (`condition_to_device`/`integrate_translation` did `np.asarray`
on tensors Lightning already moved to cuda) — now tensor-aware. `EnergyDistance` used plain Euclidean; cellflow's
scPerturb E-distance (Peidli2024) uses **squared** Euclidean — corrected.

**Not pursued:** `torch.compile`/bf16 on the VF — negligible once coupling is jitted (step ≈ couple 2.5 + fwd/bwd
4 ms). **Startup one-time costs** (not per-step): torch import 42 s cold on Lustre, CUDA init ~36 s, first zarr
batch 12 s, XLA jit 13 s.
