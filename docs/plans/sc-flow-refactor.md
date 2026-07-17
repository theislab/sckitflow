# sc-flow-tools & cellflow refactor (spec)

**Status:** plan-only. Part of the [roadmap](roadmap.md), **WS2** (sc-flow-tools) and **WS3**
(cellflow). Both need `binded` (WS1). The data-contract detail is in
[data-layer-separation.md](data-layer-separation.md).

---

## WS2 — sc-flow-tools (this repo)

Goal: sc-flow-tools becomes **one torch implementer** that consumes `binded` for data and keeps
methods/loops internal. Three tracks, roughly in order.

### Track A — data: consume `binded`
- Add dependency on `binded`; construct a `FlowProblemSpec` and use `build_loader(adata)` where the
  old sampler/index path was used.
- **Retire** `DataManager.compile_data`: `_get_mapped_index`→`MappedLevelIndex`,
  `_get_matched_distributions`, `_get_distribution_data` (streaming replaces in-memory indices).
- The schemas themselves move to `binded` (see [binded-spec.md](binded-spec.md) "What moves"). If
  `DataManager` survives, it becomes a thin holder of a `ProblemSpec` + loader; likely replaced by
  `FlowProblemSpec` directly.
- Point `trainer/_trainer.py` at batches coming from `DAGLoader`.

### Track B — torch-only collapse (the "too many ids" fix)
- Make `backends/jax/**` retirement a tracked follow-on (cellflow is the jax path). Until removed,
  stop threading `backend=` through new code.
- Drop the `backend` parameter from `config/_resolve.py` registries
  (`_PROBABILITY_PATH_REGISTRIES[backend]`, `_FLOW_SOLVER_REGISTRIES[backend]`) and the per-backend
  method trees — one backend, no dispatch. This removes the bulk of the interface-id proliferation.
- `MethodCapabilities.backends` collapses to `{"torch"}` (or drops).

### Track C — methods/model stay internal (no cross-lib contract)
- Keep `methods/` (`register_method`, `category="flow"|"general"`, `step_fn`), the composable axes
  (coupling/path/vf/solver), `RunConfig`, capabilities — all **internal** to sc-flow-tools. Do **not**
  promote them to `binded`.
- Optional simplification (recorded, not required): collapse `flow`/`general` categories into "declare
  optional pieces (`path`/`coupling`/`solver`) + a custom `step_fn`" — matches the "loss is custom,
  not categorized" intent.

### Explicitly deferred here
- `Method.fit()` / per-method loop abstraction and the **ODE inverse-problem** loop
  (train-through-ODE method vs frozen-field posterior op — fork unresolved).
- The 80 stale-test-double failures (separate cleanup; see [data-layer-separation.md](data-layer-separation.md) §2).
- Already-done env/CI work (py3.11, extras split) — do not redo.

### Rough order
Track A (data) → Track B (backend collapse) can proceed alongside → Track C is opportunistic.

---

## WS3 — cellflow

Goal: cellflow consumes `binded` and keeps everything else (jax/optax loop, solvers, networks).

- Replace the **vendored** `src/cellflow/data/dagloader` and `from dagloader import ...` with a
  dependency on `binded` (`from binded import DAGLoader, SamplerConfig, Scheme, ...`).
- Verify cellflow consumes `FlowProblemSpec.build_loader(...)` output unchanged — it already reads
  `{source, target, condition}` (see `cf-train/src/myapp/train.py` calling
  `CellFlowAnnbatch.prepare_data`). Expect near-zero model change.
- **No** adoption of sc-flow-tools' methods/loops/RunConfig. cellflow stays an independent peer;
  "complying with the contract" = consuming `binded` batches.
- cf-train updates its one import (`dagloader` → `binded`) and, in WS5, builds the `ProblemSpec` from
  config rather than bespoke `build_source` wiring.

### Risk / watch
- cellflow currently vendors dagloader — confirm the vendored copy hasn't diverged from
  `/projects/dagloader` before swapping to `binded` (diff them; fold any cellflow-only fixes into
  `binded`).
- Keep cellflow's jax loop and PRNG threading entirely as-is.
