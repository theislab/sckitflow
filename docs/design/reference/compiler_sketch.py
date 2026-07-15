"""
sc_flow  ->  dagloader   :  a COMPILATION (lowering) pass
=========================================================

Goal: take sc_flow's high-level spec (a `DataManager` config over one or more sources)
and LOWER it to dagloader IR — `Scheme` / `Node` / `Bind` / `SamplerConfig` + a
`condition_fn` — so dagloader is the only runtime. No `DistributionData`, no `NestedData`
is ever materialized on the streaming path.

This is the direct analogue of cellflow's `cellflow/data/_annbatch.py::build_annbatch_training`,
but the FRONT-END is sc_flow's vocabulary (state / conditions / groups / coupling,
control_values_dict / matched_keys) instead of cellflow's bio one.

Framing: `DataManager.compile_adata(adata) -> NestedData` already exists (it literally
"compiles"). We add a SECOND backend for the same compiler:

        DataManager.compile_adata(adata)        -> NestedData      (in-memory, today)
        DataManager.lower_to_scheme(source)     -> CompiledScheme  (dagloader IR, new)

Both read the SAME config. The in-memory one materializes and slices; the lowering one
emits weights + binds and hands the actual reads to annbatch.

    ┌────────────────────── sc_flow front-end (obs + .uns only) ──────────────────────┐
    │ DataManager config: sample_rep, conditions{level:cols}, groups, *_reps,          │
    │                     control_values_dict | matched_keys, coupling, seed           │
    │ HierarchicalIndexer -> sort columns / hierarchy    QueryFactory -> key tuples    │
    └──────────────────────────────────┬──────────────────────────────────────────────┘
                                       │  lower_to_scheme(source, sampler_cfg)
                                       ▼
    ┌────────────────────────── dagloader IR (the "bytecode") ────────────────────────┐
    │ Scheme(stores={...}, nodes={target:Node, source:Node}, root="target",           │
    │        binds=(Bind("target","source", common=groups_cols),), seed)              │
    │  NB: "store" = a named data container (was dagloader's `sources`, renamed to     │
    │      avoid colliding with the flow source/target duality). target/source below   │
    │      are FLOW roles (the model's StepData.target_*/source_*), not stores.        │
    │ condition_fn(leaf) -> {realm: emb}      SamplerConfig(batch/chunk/preload)       │
    └──────────────────────────────────┬──────────────────────────────────────────────┘
                                       ▼   DAGLoader(scheme, cfg, condition_fn)  -> batches
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

# dagloader IR (the compile target)
from dagloader import Bind, Node, Scheme, SamplerConfig, inverse_frequency, uniform

Leaf = tuple  # one value per hierarchy column (groups... then conditions...)


@dataclass(frozen=True)
class CompiledScheme:
    """The lowered program: dagloader IR + the encoder, ready for DAGLoader/DAGEvalLoader."""
    scheme: Scheme
    condition_fn: Callable[[Leaf], dict[str, np.ndarray]]
    sampler_config: SamplerConfig | Mapping[str, SamplerConfig]
    # + whatever the model needs for construction (dims registry, feature names)


# ─────────────────────────────────────────────────────────────────────────────────────
# THE COMPILER  — eight lowering passes. Each pass maps ONE sc_flow construct to dagloader.
# ─────────────────────────────────────────────────────────────────────────────────────

def lower_to_scheme(
    dm,                                   # a configured sc_flow DataManager (front-end)
    stores,                               # AnnData | DatasetCollection | {name: Container}
    *,                                    #   ("store" = a named data container; NOT the flow source)
    sampler_config: SamplerConfig | Mapping[str, SamplerConfig],
    weighting: str = "uniform",           # "uniform" | "inverse_frequency"
    source_in_memory: bool = True,
    min_obs_per_leaf: int = 0,
    chunk_size: int = 1,
    seed: int = 0,
) -> CompiledScheme:

    # ── Pass 1 · HIERARCHY → Node.cols + the leaf key space ──────────────────────────
    # sc_flow's HierarchicalIndexer already defines the sort/hierarchy columns
    # (groups first, then each conditions level). That ordered tuple IS a dagloader
    # Node's `cols`; a leaf is a value-tuple over it.
    groups_cols = tuple(dm.indexer.groups_cols)                 # top level == the match key
    cond_cols = tuple(dm.indexer.conditions_cols)               # the conditioning levels
    cols = (*groups_cols, *cond_cols)

    # Read ONLY obs (+ .uns) — never cells. Dedup to the unique leaves: a few 1e4 rows,
    # not 1e8 cells (the same O(n_leaves) trick cellflow uses to keep prep fast).
    any_store = next(iter(stores.values())) if isinstance(stores, Mapping) else stores
    obs = _obs_columns(any_store, cols)                         # dagloader._io.obs_columns
    uniq = obs[list(cols)].drop_duplicates().reset_index(drop=True)
    all_leaves = [tuple(r) for r in uniq.to_numpy()]

    # ── Pass 2 · REPRESENTATION → Node.keys ──────────────────────────────────────────
    # sample_rep -> "X" | "obsm/<k>". A distinct coupling rep or streamed response rep
    # becomes an ADDITIONAL aligned key (dagloader streams several reps of the same rows).
    keys = _rep_key(dm.sample_rep)
    if dm.coupling_rep is not None and dm.coupling_rep != dm.sample_rep:
        keys = (keys, _rep_key(dm.coupling_rep))               # aligned (state, coupling)

    # ── Pass 3 · PARTITION → which leaves are target vs source ───────────────────────
    # This is where sc_flow's matching mode selects the two leaf sets. Two cases:
    if dm.matched_keys is not None:
        # one-to-one: an ARBITRARY source_key -> target_key function. dagloader has no
        # such primitive, so we SYNTHESIZE a shared key column and bind on it (this is the
        # trick the Bind docstring prescribes). See Pass 5.
        target_leaves, source_leaves, cols, keys, match_on = _lower_matched_keys(
            dm.matched_keys, all_leaves, cols, keys
        )
    else:
        # one-to-many: `control_values_dict` names the SOURCE value per level (cellflow's
        # control_key generalized). Split leaves by whether they carry the source values;
        # bind target->source on the group columns (same-context matching).
        src_values = dm.control_values_dict or {}
        def is_source(leaf: Leaf) -> bool:
            return all(leaf[cols.index(c)] == v for c, v in src_values.items())
        source_leaves = [lf for lf in all_leaves if is_source(lf)]
        target_leaves = [lf for lf in all_leaves if not is_source(lf)]
        match_on = groups_cols                                 # Bind.common

    # ── Pass 4 · WEIGHTS → Node.weights (the selection IS the weight) ────────────────
    # sc_flow samples leaves by inverse-frequency on target obs; dagloader carries that as
    # explicit per-leaf weights. `min_obs_per_leaf` + the chunk run-length filter
    # zero-weight untrainable / unstreamable TARGET leaves (source is never filtered).
    target_weights = _target_weights(
        any_store, cols, target_leaves,
        weighting=weighting, min_obs_per_leaf=min_obs_per_leaf, chunk_size=chunk_size,
    )
    source_weights = uniform(source_leaves)                    # source: keep all, weight-uniform

    # ── Pass 5 · BINDING → Scheme.binds ──────────────────────────────────────────────
    binds = (Bind("target", "source", common=match_on),)

    # ── Pass 6 · ENCODER → condition_fn(leaf) ────────────────────────────────────────
    # Reuse sc_flow's OWN schemas + _utils encoders over the deduped `uniq` frame, keyed
    # by leaf. Same schemas as compile_adata -> embeddings byte-identical to the in-memory
    # path (parity-testable). Produces {realm: emb} per leaf = the container's old
    # get_metadata_dict, but computed per unique leaf instead of per cell.
    condition_fn = _build_condition_fn(dm, uniq, cols)

    # ── Pass 7 · STORES → Scheme.stores (multi-input lives here) ─────────────────────
    # "store" = a named data container (dagloader's old `sources`, renamed away from the
    # flow source/target). Single input: both flow nodes read ONE store. Multi-input: the
    # target and source flow endpoints read DIFFERENT stores -> {"a": A, "b": B}, each Node
    # naming its own store. Container-agnostic: A/B are AnnData | DatasetCollection | list.
    if isinstance(stores, Mapping):
        store_map = dict(stores)
        tgt_store, src_store = "target_store", "source_store"   # or whatever keys the caller gave
        store_map = {tgt_store: stores["target"], src_store: stores["source"]}
    else:
        store_map = {"data": stores}
        tgt_store = src_store = "data"

    # node keys "target"/"source" are FLOW roles (== StepData.target_*/source_*), not stores.
    nodes = {
        "target": Node(store=tgt_store, cols=cols, keys=keys, weights=target_weights),
        "source": Node(store=src_store, cols=cols, keys=keys, weights=source_weights,
                       in_memory=source_in_memory),
    }

    # ── Pass 8 · READ PARAMS → SamplerConfig (structure-free, kept off Node/Scheme) ──
    scheme = Scheme(stores=store_map, nodes=nodes, root="target", binds=binds, seed=seed)
    return CompiledScheme(scheme=scheme, condition_fn=condition_fn, sampler_config=sampler_config)


# ─────────────────────────────────────────────────────────────────────────────────────
# Notes on the two non-obvious passes
# ─────────────────────────────────────────────────────────────────────────────────────
#
# Pass 3/5 · matched_keys (one-to-one).  sc_flow lets you pair arbitrary source group ->
#   target group. dagloader binds only by matching shared COLUMN VALUES. Since a
#   source->target map is a function, `_lower_matched_keys` tags each row with a synthetic
#   column `_match_id` (target rows get their own leaf id; the matched source rows get the
#   SAME id), adds `_match_id` to `cols`, and binds on it: Bind(common=("_match_id",)).
#   No new dagloader primitive needed — the extraction's section-6 finding, realized.
#
# Pass 3 · the hierarchical tree is FLATTENED.  sc_flow's NestedData nests groups ->
#   conditions -> leaf. dagloader wants a flat weighted partition per source + a bind.
#   The compiler does NOT walk the nested tree; it derives the flat leaf set from the
#   unique `cols` combos and lets `Node.weights` + `Bind` reconstruct the same
#   target/source pairing the nested tree encoded. (The nesting was only ever a grouping
#   convenience for in-memory slicing; it carries no information the (cols, weights, bind)
#   triple doesn't.)


# ─────────────────────────────────────────────────────────────────────────────────────
# helpers (stubs — the real ones reuse dagloader._io + sc_flow.data.schemas/_utils)
# ─────────────────────────────────────────────────────────────────────────────────────

def _rep_key(rep: str | None) -> str:
    return "X" if rep in (None, "X") else f"obsm/{rep}"

def _obs_columns(source, cols):
    from dagloader._io import obs_columns
    return obs_columns(source, list(cols))

def _target_weights(source, cols, target_leaves, *, weighting, min_obs_per_leaf, chunk_size):
    """uniform, or inverse_frequency from per-leaf obs counts; then zero-weight leaves
    below min_obs_per_leaf or with a contiguous run < chunk_size (streaming guard).
    Mirrors build_annbatch_training's leaf_codes / bincount / run-length pass."""
    if weighting == "uniform" and min_obs_per_leaf == 0 and chunk_size <= 1:
        return uniform(target_leaves)                          # byte-identical to no filter
    ...  # leaf_codes -> bincount (total) + run-length (min run); build {leaf: weight}

def _build_condition_fn(dm, uniq, cols) -> Callable[[Leaf], dict[str, np.ndarray]]:
    """Run sc_flow's ConditionDataSchema / GroupsDataSchema (+ encoders) over `uniq`,
    index the resulting reps by leaf tuple, return leaf -> {realm: emb}."""
    ...

def _lower_matched_keys(matched_keys, all_leaves, cols, keys):
    """Synthesize a `_match_id` column so an arbitrary source->target map becomes a Bind
    on shared column values. Returns (target_leaves, source_leaves, cols', keys, match_on)."""
    ...


# ─────────────────────────────────────────────────────────────────────────────────────
# THE MODEL ABI  —  what the runtime hands back to the model, and the wrap-back adapter
# ─────────────────────────────────────────────────────────────────────────────────────
#
# The compiler above produces IR. At RUNTIME `DAGLoader(scheme, cfg, condition_fn)` yields
# raw arrays per batch:  {"target": ndarray(B, d), "source": ndarray(B, d), "condition": {realm: emb}}
#
# But sc_flow methods DO NOT take arrays. They take a `MatchedDistributions`:
#
#     TorchGenerativeFlow.train_step(matched_distr: MatchedDistributions) -> dict
#     TorchGenerativeFlow.predict(matched_distr: MatchedDistributions)   -> PredictionData
#
# internally -> `_extract_step_data` -> a `StepData` with 10 fields:
#     target_state, target_coupling_lin, target_coupling_quad, target_condition_data, target_group_data,
#     source_state, source_coupling_lin, source_coupling_quad, source_condition_data, source_group_data
#
# So the adapter rebuilds a per-batch `MatchedData(DistributionData, DistributionData)` from
# the streamed arrays — only B (batch_size) rows, so it's cheap. Containers survive, but
# EPHEMERAL and tiny (per-batch), never whole-population. dagloader owns selection + reads;
# the little container is just the model's input ABI.
#
#     def dispatch_batch(batch) -> MatchedData:              # this is dagloader's condition_fn's peer
#         tgt = DistributionData(
#             state_data=StateData(batch["target"]),
#             condition_data=_reps_to_mixed(batch["condition"]),   # from condition_fn(leaf)
#             groups_data=...,                                     # from condition_fn(leaf)
#             target_coupling_data=CouplingData(batch.get("target_coupling", batch["target"])),
#         )
#         src = DistributionData(
#             state_data=StateData(batch["source"]),
#             source_coupling_data=CouplingData(batch.get("source_coupling", batch["source"])),
#         )
#         return MatchedData(target_distribution=tgt, source_distribution=src)
#
# Then the existing FTrainSampler role is replaced by:  for batch in DAGLoader(...):
#                                                            method.train_step(dispatch_batch(batch))
#
# ALTERNATIVE (bigger, cleaner long-term): teach the methods to accept `StepData` directly
# and have the adapter emit StepData — skipping the DistributionData round-trip entirely.
# That touches every backend method's `_extract_step_data`; the wrap-back above touches none.
