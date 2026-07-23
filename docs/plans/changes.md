# Protocol changes (data layer)

Terse changelog of the data-protocol refactor (schemas + `compile_obs` + binded). Replaces the
verbose plan/diagram docs (removed).

## Changes

- Stripped the in-memory containers (`Distribution`/`Coupling`/`Nested`/`Matched`Data,
  `MappedLevelIndex`, `DataManager`, indexer/selector/samplers); the repo stores no arrays.
- Cells stream via **binded**; dropped `StateData`/`MixedTypeData`. `CategoricalData` kept only as the
  per-leaf `condition_lookup` builder.
- `groups` split into `match_context` (matching only, not embedded) + `covariates` (embedded only);
  a column may be in both.
- `reps` + `encoding` merged into one `Encoder` (`transform`/`inverse_transform`); a `.uns` lookup is
  just a `lookup` encoder; the embedded set is derived from the encoder-map keys.
- `GroupsDataSchema` → `CovariatesDataSchema`.
- Coupling `source_rep`/`target_rep`/`n_shared_dims` → role refs `src/tgt_lin/quad` (`anndata.acc`
  accessor or str); regime inferred; streamed as extra aligned binded `Node` keys.
- Entry point: `compile_obs` (obs-only) → `CompiledData{scheme, condition_lookup, cols, coupling}`;
  `FlowSpec.build_loader` → `binded.Loader`.
- `control_key` is now **required, no default** (was `"control"`), and moved next to the other required
  args.
- Removed `CompiledData.data_dim`: the state feature dim is read off the streamed batch on demand;
  `compile_obs` is obs-only and never sees cells, so the field was always `None`/redundant.
- Control is a boolean `control_key` column, not string `control_values_dict` values — simpler; **can
  be reverted** if per-level control values are needed again.

## Things to do

- **Simplify the encoders.** The `Encoder` ABC + per-type sklearn wrappers are heavier than needed;
  slim the hierarchy.
- **Public ref→loc normalizer.** `_ref_loc` builds a throwaway `Node("data", ("_ref_",), ref)` to get
  binded's loc-string — replace with a proper public binded normalizer.
- **Coupling loc contract can fail.** When a coupling ref resolves to the *same* loc as the state rep,
  it is deduped out of the streamed keys but still recorded in `CompiledData.coupling`, so the model
  must resolve that role from the state rep. This implicit contract has a failing case — make it
  explicit / guard it.
