# Design: an out-of-core streaming data path for `sc_flow`

Status: **proposal** · Scope: bring cellflow's `CellFlowAnnbatch` streaming capability into
`sc-flow-tools`, expressed entirely in `sc_flow`'s own domain-neutral vocabulary
(`state` / `condition` / `response` / `groups` / `coupling`, a `tree` of `MatchedData` leaves) —
with **no** perturbation / control / covariate terminology.

---

## 1. Goal

Today `sc_flow` is **in-memory only**: schemas pull dense arrays out of an `AnnData`, the whole
population lives in a `DistributionData`, and tree leaves are numpy `slice`s. There is no lazy /
backed / chunked path (grep for `annbatch|DatasetCollection|out_of_core|zarr|streaming|backed` in
`src/sc_flow` returns nothing).

cellflow solved the same problem with a two-layer stack. We want the *same capability* here, but
landed in `sc_flow`'s vocabulary rather than cellflow's bio one.

## 2. What "the CellFlowAnnbatch interface" actually is (two layers)

1. **`dagloader`** (`cellflow/src/dagloader/`) — the streaming engine, already domain-neutral:
   - `Scheme` — a rooted tree of `Node`s over named **sources**.
   - `Node` — partitions one source's rows into **leaves** (unique combinations of `cols`) with a
     per-leaf **weight** mapping. A weight of `0` / an absent combo is *excluded* — inclusion **is** a
     positive weight (no separate `select` mask).
   - `Bind(parent, child, common)` — conditions the child leaf on the parent leaf by matching the
     `common` columns. This is the **source↔target matching**.
   - `SamplerConfig` — read params (`batch_size`, `chunk_size`, `preload_nchunks`), kept off the
     structural `Node`/`Scheme`.
   - `DAGLoader` — streams `{"target", "source", "condition"}` batches; `DAGEvalLoader` reads each
     leaf's full population for validation.
   - The engine is index-free and container-agnostic (in-memory `AnnData` *or* out-of-core
     `annbatch.DatasetCollection`). Its README explicitly names `sc-flow-tools` as a target consumer.
   - The **only** bio leakage is in its *docstrings* ("perturbed", "control", "cell line") and one
     convenience factory (`perturbation_scheme` in `_schemes.py`). The types themselves are neutral.

2. **`cellflow/data/_annbatch.py::build_annbatch_training` + `model/_cellflow_annbatch.py::CellFlowAnnbatch`**
   — the **bio wrapper**. This is the layer that carries `perturbation_covariates`, `control_key`,
   `split_covariates`, `sample_covariates`, `sample_rep`, `rep_dict`. It reads only `obs` (+ `.uns`
   embedding tables), deduplicates to the unique grouping combos, drives cellflow's `DataManager` +
   `build_condition_data` to produce per-leaf condition embeddings, and assembles a `dagloader.Scheme`
   (perturbed root, matched-control child bound on the context columns) + a `condition_fn(leaf)`.

**Porting to `sc_flow` = rewrite layer 2 in `sc_flow`'s vocabulary, reusing layer 1 unchanged.**
`sc_flow` already has the perfect neutral vocabulary; nothing new needs inventing.

## 3. Vocabulary translation

| cellflow annbatch (bio) | `dagloader` (engine) | `sc_flow` (neutral) |
|---|---|---|
| perturbed cells = **target** | `root` node | `MatchedData.target` / root of the tree |
| matched control = **source** | bound child node | `MatchedData.source` |
| `control_key` (bool flag) | which leaves are target vs child | **`control_values_dict`** (source value per level) or **`matched_keys`** |
| `split_covariates` (context) | `Bind.common` | **`groups`** (top hierarchy level) — the match key |
| `perturbation_covariates` | node `cols` | **`conditions`** (hierarchy levels) |
| `sample_covariates` | node `cols` | `conditions_covariates` |
| `sample_rep` (`"X"` / `obsm/<k>`) | `Node.keys` | **`sample_rep`** → `StateDataSchema` |
| covariate embeddings (`rep_dict` → `condition_fn`) | `condition_fn(leaf)` | `ConditionDataSchema` / `GroupsDataSchema` + `.uns` reps |
| `Scheme` (sources + nodes + binds) | `Scheme` | the **tree** (analogue of `NestedData`); leaves = **nodes** |
| `min_cells_per_condition`, run-length filter | root leaf weights | per-leaf weight filter on the target level |

The mapping is *exact* because `sc_flow` already partitions into a `HierarchicalIndexer` tree of
**groups → conditions → leaves**, and already expresses matching as `control_values_dict` /
`matched_keys` — which is precisely what `Bind` does. (The `Bind` docstring in `dagloader` already
cites both by name.)

## 4. Architecture

### 4.1 Extract `dagloader` into a shared package

Follow the **`classmap` precedent**: `sc-flow-tools` already lists `classmap` in `dependencies` as a
"shared … layer (extracted from this repo)". `dagloader` becomes the sibling shared **streaming
engine**. cellflow's `pyproject.toml` already anticipates this — its wheel config comments that
`dagloader` is "vendored for now but kept as a standalone top-level package … so it can later be
extracted into an external dependency with no code changes."

Plan:
- Move `cellflow/src/dagloader/` into its own package (`dagloader`), with cellflow's `annbatch`
  extra carried along (it pins a fork of `annbatch` by git URL — see cellflow `pyproject.toml`).
- Both `cellflow-tools` and `sc-flow-tools` depend on it (behind an optional `streaming` /
  `annbatch` extra, since it drags in `annbatch` + optionally `cupy`).
- De-bio the engine **in place** during extraction: neutralize the docstrings (`perturbed`→`target`,
  `control`→`source`, drop "cell line"), and either drop `perturbation_scheme` or rename it to a
  neutral `matched_scheme(source, *, groups, levels, source_values, key, seed)`. Types
  (`Scheme`/`Node`/`Bind`/`SamplerConfig`) are unchanged — the extraction is behavior-preserving.

### 4.2 New `sc_flow.data.streaming` module (the neutral layer 2)

Mirrors `build_annbatch_training`, but reads `sc_flow`'s config and reuses `sc_flow`'s schemas /
encoders instead of cellflow's `DataManager`.

```python
# sc_flow/data/streaming/_build.py
def build_streaming_tree(
    source,                       # AnnData | annbatch.DatasetCollection
    *,
    sample_rep,                   # -> StateDataSchema         (dagloader Node.keys)
    conditions,                   # dict{level: obs-cols}      (hierarchy levels)
    groups=None,                  # top level + the match key  (Bind.common)
    conditions_reps=None,         # dict{level: .uns table}
    groups_reps=None,
    control_values_dict=None,     # source value per level     -> Bind
    matched_keys=None,            # explicit source->target    -> tag + Bind
    seed=0,
    source_in_memory=True,        # materialize the source node into RAM (dagloader Node.in_memory)
    min_obs_per_leaf=0,           # zero-weight tiny target leaves
    chunk_size=1,                 # drives the run-length weight filter
) -> StreamingData: ...

@dataclass(frozen=True)
class StreamingData:
    tree: Scheme                                      # dagloader.Scheme (target root, source child)
    condition_fn: Callable[[Leaf], dict[str, np.ndarray]]
    dims_registry: DataDimensionalitiesRegistry       # for model construction
    # (+ whatever compile_adata records today: encoders, feature names)
```

Reuse, not reinvention:
- **`condition_fn`** is built exactly as cellflow builds it, but from `sc_flow`'s own machinery:
  deduplicate `obs` to the unique `(groups, conditions)` combos (a few ×10⁴ rows, **not** the ~10⁸
  cells — this is the O(n_leaves) trick that keeps prep fast), run `ConditionDataSchema` /
  `GroupsDataSchema` + the `_utils` encoder factories over the deduped frame, and index the result by
  leaf. Because it reuses the same schemas as the in-memory path, embeddings are **identical** to
  `register_adata` — parity-testable.
- **The tree** is a `dagloader.Scheme`: root node = target leaves (positive weight = the target
  selection), child node = source leaves, `Bind(target, source, common=groups_cols)` for
  same-group matching. `control_values_dict` selects a specific source value per level; `matched_keys`
  (an arbitrary source→target function) is handled the way the `Bind` docstring prescribes — tag each
  side with a shared key column and bind on it.

### 4.3 `SCFlow` streaming entry point + sampler adapter

Parallel to the existing `register_adata` / `train` / `predict`, keeping the two-phase
class-attribute pattern:

```python
SCFlow.register_streaming_adata(source, *, sample_rep, conditions, groups=None,
                                control_values_dict=None, matched_keys=None, ...)
# train() gains an FStreamingTrainSampler that wraps dagloader.DAGLoader and yields the SAME
# StepData-shaped batches the torch/jax methods already consume — so methods are UNCHANGED.
```

**Batch contract.** `dagloader` yields `{"target", "source", "condition"}`. The adapter reshapes
that into the method's `StepData` fields
(`target_state` / `source_state` / `target_condition_data` / `target_group_data` / `source_*`, plus
the coupling terms). Coupling defaults to the state rep (as in the in-memory path); a distinct
`coupling_rep` maps to a second aligned `Node.keys` entry (dagloader streams several aligned reps of
the same sampled rows). Validation reuses `DAGEvalLoader` behind an `FValidationSampler`-shaped
adapter (read each held-out leaf's full population + matched source).

### 4.4 Weight filters (target-leaf only)

Direct analogues of cellflow's two perturbed-only filters, renamed:
- `min_obs_per_leaf` — zero-weight any **target** leaf with fewer than this many total observations.
- `chunk_size > 1` run-length filter — annbatch reads contiguous `chunk_size` slices, so every run of
  a positive-weight class must be `≥ chunk_size`; target leaves whose smallest contiguous run is
  shorter are zero-weighted (and logged). With both inactive (`min_obs_per_leaf=0`, `chunk_size=1`)
  target weights are `uniform` — byte-identical to no filtering. The **source** node is never filtered
  (materialized+sorted in RAM when `source_in_memory`, else annbatch's own concern).

## 5. Why reuse `dagloader` rather than reimplement

- It is *designed* to be shared (README + the `Bind` docstring already speak `sc_flow`'s
  `control_values_dict` / `matched_keys`).
- The index-free weighted `ClassSampler` + `BoundClassSampler` matching is non-trivial to
  re-derive correctly; a second implementation would drift.
- The `classmap` extraction is a proven precedent in this exact repo.

Trade-off: it adds an `annbatch` (fork) dependency behind an extra. Acceptable — streaming is
opt-in, and the in-memory path keeps zero new deps.

## 6. Open questions

1. **`Scheme` vs `NestedData`.** `dagloader.Scheme` (weighted, index-free) and `sc_flow`'s
   `NestedData` (slice-indexed) are two different trees. Do we (a) keep them distinct and only convert
   at the batch boundary (proposed), or (b) unify — e.g. give `classmap.MappedTree` a weighted
   variant so both engines share one tree type? (a) is lower-risk for a first cut.
2. **Coupling over incomparable spaces** (`n_shared_dims`, lin/quad). The in-memory path supports
   Gromov-OT coupling; streaming needs the second aligned rep wired through `Node.keys`. Land the
   linear (shared-space) case first, quad later.
3. **`response` / target covariates.** cellflow's annbatch path has no `response` analogue. Decide
   whether streaming supports `ResponseDataSchema` in v1 or defers it.
4. **Where `dagloader` lives** — its own repo, or a `packages/dagloader` inside a monorepo alongside
   `classmap` (there is a `CellFlow2/packages/classmap` today).

## 7. Suggested rollout (small, reviewable increments)

1. Extract + de-bio `dagloader` into a standalone package; wire it as an optional dep in both repos.
   No `sc_flow` behavior change yet.
2. `sc_flow.data.streaming.build_streaming_tree` + `StreamingData`, reusing `sc_flow` schemas for
   `condition_fn`. Parity test: embeddings identical to `register_adata` on a toy dataset.
3. `FStreamingTrainSampler` adapter → `StepData`; a smoke train over an in-memory `AnnData` streamed
   through `dagloader` (no zarr yet) that matches the in-memory sampler's batch shapes.
4. Out-of-core: same toy written to a zarr `DatasetCollection`; assert identical batch shapes/dtypes
   and a short train step.
5. `SCFlow.register_streaming_adata` + validation via `DAGEvalLoader`; weight filters
   (`min_obs_per_leaf`, `chunk_size`).
6. Docs + a notebook mirroring `docs/notebooks/data_preparation.ipynb` for the streaming path.

## 8. Finalized decisions (this design round)

The design converged on a stricter **"labels only, no data"** principle. The refinements below
supersede the more conservative phrasing in §4 where they conflict.

1. **The streaming spec holds labels + relations, never rows or arrays.** A leaf is a *label
   tuple*; a relation is a *bind on shared label columns*; a weight is a *per-label count*; a
   representation is a *location* (`Node.keys`), not an array. Compilation reads only `obs` (+ `.uns`
   embedding tables) and *store handles* — it never touches cells. This is dagloader's `Scheme`
   verbatim (its README: "no row indices are exposed").

2. **`store`, not `source`, for containers.** "source" is reserved for the flow endpoint
   (`StepData.source_state`). dagloader's container concept (`Scheme.sources` / `Node.source`) is
   renamed to `Scheme.stores` / `Node.store` as part of the extraction, removing the three-way
   collision. The two flow **nodes** are still named `target` / `source` (they *are* the flow roles).

3. **No state/coupling containers on the streaming path — `StepData` is the single ABI.**
   `StateData` (a 1-field array wrapper) and `CouplingData` (two arrays + a column split) are pure
   ceremony here: `_extract_step_data` already unwraps them to the plain `StepData` tensors. So the
   runtime adapter maps an annbatch batch **straight to `StepData`**; `_extract_step_data` is deleted
   on this path. Arrays exist **only** as the transient annbatch batch, owned by annbatch. Kept
   containers: `CategoricalData` / `MixedTypeData` — they do real work (obs→`.uns` lookups) and become
   the `condition_fn`. `DistributionData` does not exist on this path (it *is* the materialized
   population — the thing we're replacing); its label role → `Node.cols`, its metadata role →
   `condition_fn(leaf)`.

4. **Coupling = a representation choice + a split choice, both label-level.**
   *Which* rep to match in → a second (per-node) entry in `Node.keys`, so target and source may match
   in different reps (incomparable spaces). *How* to split lin/quad → **preferably two separate whole
   reps** (`obsm["coupling_lin"]`, `obsm["coupling_quad"]`) referenced as two keys — no scalar, no
   per-batch slice; the `n_shared_dims` scalar (column-slice at the `StepData` boundary) is the
   fallback when you won't duplicate storage. `assert_same_spatial_dims` becomes automatic.

5. **anndata 0.13 `acc` accessors: use for *locations*, not for column slicing.** Verified against
   anndata 0.13.0: `A.obsm[k][:, i]` (single column) and `[:, [i,j]]` (→ *list* of column refs) work;
   `A.obsm[k][:, :n]` (contiguous range) is **explicitly rejected** (`process_idx` allows only
   int / int-list). Also, dagloader's `_rep_loc` currently discards the index (keeps only `dim`/`k`),
   and annbatch reads whole reps. So `A.obsm[k][:n]` is a dead end today. Do adopt accessors as typed,
   MuData-aware, typo-catching **location keys** (`A.obsm["X_pca"]` instead of `"obsm/X_pca"`) —
   dagloader already accepts `RepKey = str | RefAcc`. Express the coupling split as two reps (point 4),
   not an accessor slice.

### 8.1 Compilation pipeline (`DataManager.lower_to_scheme`)

`compile_adata` (in-memory) and `lower_to_scheme` (streaming) are **two backends of one compiler**
over the same config. Eight obs-only passes:

| Pass | sc_flow construct | dagloader object |
|---|---|---|
| 1 Hierarchy | `HierarchicalIndexer` sort columns (groups → conditions) | `Node.cols` + leaf key space |
| 2 Representation | `sample_rep` (+ coupling reps) | `Node.keys` (aligned tuple, whole reps) |
| 3 Partition | `control_values_dict` / `matched_keys` | which leaves are target vs source |
| 4 Weights | `uniform`/`inverse_frequency` + `min_obs`/chunk run-length | `Node.weights` (selection = weight) |
| 5 Binding | same-context match / explicit pairs | `Bind(target, source, common)` |
| 6 Encoder | `ConditionDataSchema`/`GroupsDataSchema` + `_utils` | `condition_fn(leaf) -> {realm: emb}` |
| 7 Stores | one or many `AnnData`/`DatasetCollection` | `Scheme.stores={name: Container}` |
| 8 Read params | batch/chunk/preload | `SamplerConfig` |

Two non-obvious passes: **Pass 3** flattens the hierarchical `NestedData` into a flat weighted
partition + bind (the nesting carried no info the `(cols, weights, bind)` triple doesn't); **Pass 4**
is the only pass that scans `obs` in physical order (per-leaf count + smallest contiguous run) —
still obs-only, zero cells.

Runtime: `DAGLoader(scheme, cfg, condition_fn)` streams `{target, source, condition}` → an adapter
maps it straight to `StepData` (with the coupling split from point 4) → `method.step(StepData)`.

### 8.2 Prototype (runnable, verified)

A working prototype of the compiler + runtime adapter lives in scratch
(`scflow_compile_prototype.py`). It compiles a `Spec` (groups=`cell_line`, conditions=`drug`,
`control_values={"drug":"ctrl"}`, `inverse_frequency`, `min_obs_per_leaf=5`, `chunk_size=2`) from a
toy `AnnData`'s **obs only** into a real `dagloader.Scheme` + `condition_fn`, streams it via
`DAGLoader`, and emits `StepData` with no containers constructed. Verified:
- `(A, d2)`=4 cells zero-weighted by `min_obs_per_leaf=5`; the rest carry inverse-frequency weights.
- `condition_fn(leaf)` returns label→vector (no rows).
- Source is drawn from the **same `cell_line`** as target every batch (the `Bind`, checked by
  stamping the group code into the state and asserting equality per batch).

Resolves open questions §6.1 (keep `Scheme` distinct; convert only at the batch boundary) and
§6.2 (coupling via two reps preferred, `n_shared_dims` scalar as fallback). §6.3 (`response`) and
§6.4 (where `dagloader` lives) remain open.

## 9. Full `DataManager` spec surface → lowering destination

The compiler must account for **all 28** `DataManager.__init__` params, not just the structural core.
They fall into four lowering destinations. Only ①②③ touch the compiled `Scheme`; ④ is a pure
per-batch function at the `StepData` boundary (never in the `Scheme`).

**① Scheme structure** — labels + binds + weights (the only things that define leaves/matching):

| spec | role |
|---|---|
| `conditions` | condition levels → `Node.cols` (condition part) |
| `groups` | top level → `Node.cols` (groups part) + `Bind.common` |
| `control_values_dict` | source value per level → partition + `Bind` |
| `matched_keys` | explicit source→target → `_match_id` tag + `Bind` |

**② `Node.keys` aligned reps** — per-**cell** arrays streamed alongside state (dagloader streams
several aligned reps of the same rows):

| spec | role |
|---|---|
| `sample_rep` | state rep (primary key) |
| `source_rep` | distinct rep for the source node (incomparable-space coupling) |
| `conditions_covariates` | **continuous, per-cell** condition covariates → aligned rep (NOT a per-leaf embedding) |
| `target_continuous_covs` | **response** continuous covariates (per cell) → aligned rep |
| `n_shared_dims` | lin/quad column split of the coupling rep (scalar; see §8 point 4) |

**③ `condition_fn` / encoders** — per-**leaf**, built from `obs` + `.uns`:

| spec | role |
|---|---|
| `conditions_reps`, `groups_reps` | `.uns` embedding tables → per-leaf lookup |
| `groups_encoding` | label / one-hot / functional encoder per group col |
| `target_categorical_covs_dict` | **response** categorical covariates + encoder ids → per-leaf reps |

**④ Per-batch transforms / model-side** — pure functions on the streamed tensors at the boundary:

| spec | role |
|---|---|
| `state_transform`, `state_preproc_repr_name` | preprocess the streamed state |
| `state_encoder_context` / `state_decoder_context` | external (e.g. VAE) encoder/decoder on state |
| `condition_covariates_transform_dict` / `_encoder_context_dict` / `_decoder_context_dict` | transforms/external encoders for continuous condition covariates |
| `groups_encoding_transform_fn` / `_inverse_transform_fn` | functional encoders for groups |
| `allow_paired_settings_on_condition_view` | flag for the condition-as-state view |

Two things this surfaces that the earlier sections omitted:

1. **`response`** (`target_categorical_covs_dict`, `target_continuous_covs`) is a full channel — the
   `StepData` response path. Categorical → `condition_fn`; continuous → an aligned `Node.keys` rep.
   (Was §6.3, now mapped rather than deferred.)
2. **Continuous vs categorical conditioning is a hard fork.** Categorical conditions are *per-leaf*
   (→ `condition_fn`); `conditions_covariates` are *per-cell continuous* → they must ride as an
   **aligned `Node.keys` rep** (cellflow's "state plus a per-cell continuous condition"). They cannot
   collapse into a leaf embedding.

## 10. Verification (runnable, real engine)

`reference/verify_scflow_to_scheme.py` (next to this doc) compiles an `SCFlowSpec` — written in
`sc_flow` vocabulary (`sample_rep` / `groups` / `conditions` / `control_values_dict` |
`matched_keys`) — into a **real** `dagloader.Scheme`, then drives the **real** `DAGLoader` and asserts
the streamed cells obey the compiled structure. **16/16 checks pass** in cellflow's venv (real
`dagloader` + `annbatch` + anndata 0.13):

- **Structural (obs/labels only, zero cell reads):** `cols` = groups→conditions; `keys` = location
  string; target leaves = non-control combos; source leaves = control combos;
  `Bind(target, source, common=groups)`; uniform weights.
- **Runtime (real `DAGLoader`):** 20/20 batches — target class-coherent and ≠ control, source =
  control, and **source group == target group every batch** (the `Bind` match, verified by encoding
  group/condition ids into the streamed rep and asserting equality); `condition_fn` one-hot aligns
  with the streamed target condition.
- **`matched_keys` (one-to-one):** the synthesized `_match_id` tag+`Bind` lowering compiles correctly.

Caveat: `condition_fn` there is a one-hot **stand-in** for `sc_flow`'s real
`ConditionDataSchema`/`GroupsDataSchema` (Pass ③). Wiring Pass ③ to the real schemas needs an
environment with both `sc_flow` and `dagloader` installed.
