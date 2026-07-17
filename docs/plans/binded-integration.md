# Task handoff — make sc-flow-tools' data layer use `binded`; drop array storage

**For:** the next agent, cold. **Status:** ready to implement. **Branch:** `experiment/binded-vocab-strip`
(merges to `feat/refactor` later). **Commits:** **no Claude/Co-Authored-By tag; 3–4 word messages**
(e.g. `use binded loader`, `drop state container`).

## One-line goal

Let **`binded`** own all array handling (streaming); this repo keeps only labels/spec + the per-leaf
condition builder. Delete the data structures that still store arrays.

## What this chat decided (context)

Goal is to express cellflow's model-init in sc-flow-tools' **own** vocabulary (no `split_covariates`/
`sample_covariates`). Design captured in [changes.md](changes.md) + [schema-generalization.md](schema-generalization.md):
1. Split matching (`match_context`) from embedding (`covariates`) — `groups` no longer does both.
2. Merge `_reps` + `_encoding` into one `Encoder` (transform/inverse); a `.uns` lookup is just an encoder.
3. Coupling → role-named refs `src/tgt_lin/quad` (`anndata.acc` `AdRef`); lin/quad inferred; no `n_shared_dims`.
4. The in-memory containers bifurcate: **labels → loader spec, arrays → streamed batch**. `CouplingData`
   (pure arrays) fully dissolves; the others keep only their label half.

The **strip already happened** (commit `b75bf80`, "strip in-memory aggregate/matching stack"): removed
`DistributionData`/`CouplingData`/`MatchedData`/`NestedData`/`DataManager`/`_dims_registry`/`grouping/`/
`samplers/`/`_coupling_data_schema`/`MappedLevelIndex`. **This task is the next step.**

## Current as-built (what imports today)

- **schemas/** `StrictDataSchema` → `State`/`Condition`/`Groups`/`Response`DataSchema (still cellflow-ish
  names — renames of Change 1–3 are **NOT applied yet**, deferred).
- **containers/** `BaseData` → `StateData`(`.X`) / `CategoricalData` / `MixedTypeData` — the array holders.
- **`data/_compile_obs.py`**: `compile_obs(adata, *, state, condition, groups, control_key,
  split_covariates) → CompiledData{scheme, condition_fn, cols}`. Reads **obs/uns only** (no cells).
  Currently does `from dagloader import Bind, Node, Scheme, uniform` — **dagloader is not installed**.

## `binded` (the target loader)

At `/Users/selman/projects/binded` (`name = "binded"`, not installed in `.venv`). Exposes
`Scheme, Node, Bind, SamplerConfig, Weights, Container, uniform, Loader, EvalLoader, …`.

**Contract — exactly what `compile_obs` produces:**
```python
from binded import Loader
loader = Loader(compiled.scheme, sampler_config, compiled.condition_fn)
for batch in loader:      # {"source", "target", "condition"} — one condition per batch
    ...
```
`Loader.__init__(scheme, sampler_config, condition_fn=None)`; `condition_fn: (leaf_tuple) →
np.ndarray | {name: np.ndarray}` — **`compile_obs.condition_fn` matches this signature verbatim.**
binded streams all cell reps via annbatch (`annbatch 0.2.0` is present); no obs/arrays materialized in
this repo.

**⚠ verify before running:** binded's `Node` field is **`keys`** (plural: a rep loc-string *or an
accessor*, or several), not a single `key`. `compile_obs` passes `Node("data", cols, key, weights)`
positionally → maps to `source, cols, keys, weights`; confirm the 4th field is `weights` and that
`keys` accepts the `"obsm/<rep>"`/`"X"` loc-strings `_sample_rep_to_key` emits. (binded `Node.keys`
accepting an *accessor* is the natural home for Change 3's `AdRef` later.)

## The task (concrete)

1. **Depend on binded.** Add `"binded"` to `pyproject.toml` `dependencies`; install editable
   (`uv pip install -e /Users/selman/projects/binded` or the repo's install flow).
2. **Swap the import** in `_compile_obs.py`: `from dagloader import …` → `from binded import …`.
   Fix any `Node`/`Scheme` signature drift (see ⚠ above).
3. **Smoke-test** end-to-end: tiny in-memory adata → `compile_obs(...)` → `binded.Loader(...)` → pull
   one batch; assert `{source, target, condition}` shapes and that **no whole-dataset array is stored in
   any repo container**.
4. **Drop array storage.** The streaming path stores no whole-dataset arrays already, so remove the dead
   array holders + their producers:
   - Delete `StateData` and `StateDataSchema.get_data`→`StateData` (state is 100% binded's — the schema
     only contributes `sample_rep` as the `Node` key).
   - Delete `MixedTypeData` + the `get_data` array-materializers on `Condition`/`Response` schemas
     (not on the `compile_obs` path).
   - **Keep `CategoricalData`** — but only as the **per-leaf** condition builder used inside
     `condition_fn` (`from_pandas` + `extract_reps` on a 1-row frame). It must not materialize
     whole-dataset arrays.
5. Keep the **data layer importable in isolation** (as `b75bf80` did). Downstream
   (`model`/`methods`/`backends`/`preprocessing`/`trainer`) still references removed symbols **by
   design** — that breakage is the binded-rewiring worklist, not this task.

## Deferred (do NOT do here)

The Change 1–3 renames (`match_context`, `Encoder` merge, `CouplingRefs`/`AdRef`) — land them *after*
binded works, on `compile_obs` + the schemas. See [changes.md](changes.md).

## References

- Design: [changes.md](changes.md), [schema-generalization.md](schema-generalization.md).
- Diagrams (live viewer symlinks `scratchpad/diagram.mmd` → these; served on `:8747`):
  [current-state.mmd](current-state.mmd) (as-built), [streaming-target.mmd](streaming-target.mmd)
  (aspiration), `schema-generalization.mmd` (pre-strip, historical).
- Roadmap: [sc-flow-refactor.md](sc-flow-refactor.md) Track A (this is that work).
