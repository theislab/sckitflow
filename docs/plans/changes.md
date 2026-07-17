# Changes — schema generalization (concise)

Goal: describe cellflow's model-init concerns in sc-flow-tools' *own* language. Detail +
diagram in [schema-generalization.md](schema-generalization.md). Native vocab only (no
`split_covariates`/`sample_covariates`).

> **Status: Changes 1–3 IMPLEMENTED** on `experiment/binded-vocab-strip`, adapted to the post-strip
> binded world (there is no `DataManager`/`HierarchicalIndexer` anymore — the changes landed on
> `compile_obs` + the `schemas/`). Commits: `e10ec71` (1 — `match_context`), `ec6dc39` (2 — one
> `Encoder` map + `GroupsDataSchema`→`CovariatesDataSchema`), `956318c` (3 — `CouplingDataSchema`
> role-refs, `anndata>=0.13` pinned for `anndata.acc`). Change 4 (container dissolution) already rode
> the binded strip (`0ae814e`). Decisions taken: embedded axis named `covariates`; one `{col: Encoder}`
> map (both directions bundled); coupling refs adopt `anndata.acc` accessors (str also accepted).
> Not done: the model-side consumption of coupling role-refs (downstream is still being rewired).

| # | Change | Why |
|---|---|---|
| 1 | **Split `match_context` from `covariates`.** `groups` no longer feeds the indexer *and* the encoder. `match_context: list[str]` drives the hierarchical split (matching only); `covariates` are embedded only. A column may be in both. No `match_context or groups` fallback. | `groups` did double duty — you couldn't "match on `cell_line` but not embed it" (or the reverse). The fallback would silently re-fuse the two axes we're separating. |
| 2 | **One `Encoder` abstraction** (`transform`/`inverse_transform`) replaces `_reps` + `_encoding` + the two `_fn` dicts. `.uns` lookup becomes just `lookup(uns_key)`; embedded set is *derived* from the encoder-map keys. Reused by conditions + response + covariates. | `reps` vs `encoding` is a false split — both are "fit a transform on a column, map each cell to a vector"; a lookup is a transform whose params come from `.uns`. Collapses 4 parallel dicts → 1; makes "what's embedded" a single source of truth. |
| 3 | **Coupling → role-named references, lin/quad inferred.** `source_rep`/`target_rep`/`n_shared_dims` → `src_lin`/`src_quad`/`tgt_lin`/`tgt_quad` (`anndata.acc` `AdRef`, or `str`). The names are **role tags**, not just locations: they tell binded which keys to stream *and how to tag them*, and the model reads by role. Regime inferred from which are present (`*_quad` ⇒ quadratic/GW). No slicing; reps arrive pre-separated. | One abstraction ("match `srcs` to `tgts`") covers linear + quadratic/GW. The role tags travel **spec → batch → model** (Wasserstein term ← `lin`, GW term ← `quad`). `adata[ref]` replaces `_extract_array`; `ref in adata` replaces presence checks. `n_shared_dims`-slicing disappears. (Refs need anndata ≥ `anndata.acc`; env has 0.12.9.) |
| 4 | **Coupling *container* dissolves; its *declaration* survives; the other in-memory containers bifurcate.** The `CouplingData` **container** (whole-dataset arrays + `init_from_state_data`/`concat`/slice) → gone; the **declaration** (the 4 role-named refs of Change 3) stays as the batch-key contract binded loads. `DistributionData`/`NestedData`/`MatchedData`/`MappedLevelIndex` split: **labels → loader spec** (`Nodes`/`Bind`/`Weights`), **arrays → streamed batch**. | OT coupling is per-minibatch — nothing whole-dataset to materialize, so the *storage* goes. But `state_lin`/`state_quad` also encode a **role** (which cost term), and that role is load-bearing: it *is* the spec that tells binded what to stream. The other containers likewise keep their *label structure* (subpopulation keys, source↔target pairing, membership) as the loader spec; only array halves stream. Matches [sc-flow-refactor.md](sc-flow-refactor.md) Track A. |

## Rename map (native)

| old | new | status |
|---|---|---|
| `split_covariates` (compile_obs param) | `match_context` | **done** (`e10ec71`) |
| `GroupsDataSchema` + `groups_reps`/`groups_encoding` | `CovariatesDataSchema` + `covariate_encoders` | **done** (`ec6dc39`) |
| `_reps` / `_encoding` / `_fn` dicts | `Encoder` (one `{col: encoder}` map, two directions) | **done** (`ec6dc39`) |
| `source_rep`/`target_rep`/`n_shared_dims` | `src_lin`/`src_quad`/`tgt_lin`/`tgt_quad` (`anndata.acc` accessor or str) | **done** (`956318c`) |

## Sequencing (as executed)

- Changes **1–2** landed with no anndata dependency.
- Change **3** adopted `anndata.acc` — `anndata>=0.13` is now pinned in `pyproject.toml`.
- Change **4** rode the `binded` streaming migration (`0ae814e`); `CouplingData`/`DistributionData`/etc.
  were removed in the earlier strip (`b75bf80`).
