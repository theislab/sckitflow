# Consolidated codebase — review ledger

Tracks the `scfit` (ML toolbox + streaming) / `sc_flow` (flow-matching) split after binded was absorbed
into `scfit.data`. **Approval rule (set by the maintainer):** only what came from **binded** is approved
as-is. Every other class/function must be reviewed one by one — check it off here once it has passed review.

Legend: ✅ approved (binded, keep as-is incl. docstrings) · ⬜ pending review · 🗑 flagged dead/duplicate.

---

## ✅ Approved — `scfit.data` (absorbed binded streaming layer)

Kept verbatim from `theislab/binded`; docstrings retained.

- `scfit/data/_loader.py` — `Loader`
- `scfit/data/_eval_loader.py` — `EvalLoader` (`control_populations`, `iter_conditions`)
- `scfit/data/_schema.py` — `Node`, `Bind`, `SamplerConfig`, `Scheme` (`from_paths`); `uniform`, `frequency`, `inverse_frequency`
- `scfit/data/_split.py` — `split_scheme`, `split_assignment`, `resolve_split_configs`
- `scfit/data/_io.py` — `key_backings`, `materialize_node`, `leaf_codes`, `obs_columns`, `load_backed_adata`, `open_source`
- `scfit/data/_backend.py` — annbatch adapter (internal)

---

## ⬜ Pending review — `scfit` (ML toolbox)

### `scfit/_component.py`
- ✅ class `ComponentSpec`
- ✅ class `ComponentRegistry` (`register`, `validate`, `parse`, `build`, `family`, `types`) — reviewed;
  `register` reworked to **dual-mode**: decorate a config dataclass that owns `build(self, context)`
  (preferred — infers `config_type`, no separate factory) *or* pass `config_type`+`build` directly.

### `scfit/_types.py`
- ✅ `NestedLayersDict` removed (no live source consumer).
- 🗑 `LayersDict` remains only to type the live transitional `init_module_from_dict`; delete both together
  when the `net` specs replace dictionary-built MLPs, then remove `_types.py` if empty.

### `scfit/_utils.py`  (single owner; `sc_flow` imports from here)
- ✅ removed the dead signature-introspection branch: `check_type_against_generic`,
  `get_fn_args_names_and_types`, `get_fn_kwargs_names_and_types`, `verify_fn_args`, and
  `verify_fn_signature`; deleted stale `tests/test_utils.py`.
- ✅ `verify_fn_kwargs_dictionary` and its private reflection helper are now removed too:
  `Resnet1dCombiner.resnet_kwargs` was replaced atomically by the typed nested `scfit.resnet` spec.
- ⬜ keep/review `check_sequence_query_against_reference` separately — it has three live consumers
  (condition-schema validation and set-encoder config/runtime key checks), so it is not an orphan.

### `scfit/nn/`
- ✅ `_modules.py` — G1–G3 landed: `BaseModule`, its abstract `_make_modules` hook, constructor-signature
  sniffing, runtime-module bookkeeping, per-leaf persistence mixin, and later `object.__setattr__` repairs
  are removed. `MLP` and `Resnet1d` are ordinary `torch.nn.Module`s with local build helpers.
  🗑 `FunctionalModule` is flow-only in practice (one time-feature wrapper); replace it with the velocity
  runtime's configured callable and remove it from the generic package.
- ✅ `_net.py` — the approved `scfit.resnet` slice of G4: `NET_REGISTRY`, `NetSpec`, and `NetContext`;
  `ResnetConfig` carries typed/validated architecture hyperparameters while its consuming slot supplies
  input, output, and conditioning dimensions.
- ✅ `_activation.py` — keep `resolve_activation`; removed orphaned `activation_id` and its reverse map.
- 🗑 `_utils.py` — delete `init_module_from_dict` + `_resolve_dim` when `scfit.mlp` lands. The helper is
  currently live, but every consumer is one of the dictionary-built MLP slots replaced by the `net`
  family (embedders, decoder, set-encoder heads, and feature-realm projection).

### `scfit/metrics/_metrics.py`
- ⬜ *partially reviewed*: `MeanAggregatedRSquared`, `EnergyDistance`, `MaximumMeanDiscrepancy`,
  `PredictionDispersion`; `rbf_kernel_torch`; `METRICS_REGISTRY` (`e-dist`, `mmd`,
  `mean_aggregated_r_squared`, `mse`, `mae`, `pred_std`/`pred_mean_abs`). **`PredictionDispersion` still
  needs a look** (partially reviewed — exercised by the smoke as a monitor metric, but its correctness/API
  not yet signed off).

### `scfit/training/`
- ✅ `_harness.py` — `TrainingModule` — reviewed & **genericized**: training-only now (`training_step`,
  `configure_optimizers`, `prog_bar_metrics` to select progress-bar keys) with a vocabulary-free no-op
  `validation_step` that just opens the Lightning val loop. The perturbation source→target /
  identity-baseline scoring moved OUT to `sc_flow.flow.PerturbationValidationCallback`; the base carries no
  `source`/`target`/metric vocabulary.
- ⬜ `_objective.py` — `Objective` (`compute_loss`), `register_objective`, `build_objective`, `OBJECTIVE_REGISTRY`
  — *discussed, deferred*: kept as-is for now (3 impls selected by a user-facing `objective=` name, so the
  registry earns more than the predictor's did); its endgame is `ComponentRegistry`, to migrate alongside
  the predictor when portable training-config / entry-point plugins land.
- ✅ `_predictor.py` — `Predictor` (`predict`, `predict_with_aux`) — reviewed; dropped
  `register_predictor`/`build_predictor`/`PREDICTOR_REGISTRY` (one impl, hardcoded name — the registry
  earned nothing; a predictor is chosen by the toolbox that owns the model, so `ODEPredictor` is
  instantiated directly). Added optional `predict_with_aux(model, batch) -> (pred, aux)` for
  trajectories/stats.

---

## ⬜ Pending review — `sc_flow` (flow-matching toolbox)

### `sc_flow/data/` (perturbation data layer — compiles onto `scfit.data`)
- ⬜ `_spec.py` — `FlowSpec` (`compile`, `build_loader`)
- ⬜ `_compile_obs.py` — `CompiledDims`, `CompiledData`, `compile_obs`
- ⬜ `_encoders.py` — `Encoder`, `OneHot`, `Label`, `Functional`, `Lookup`; `one_hot`, `label`, `lookup`, `functional`
- ⬜ `_abc.py` — `Distribution`, `MatchedDistributions`, `DataTree`
- ⬜ `_mixins.py` — `MappedTree`, `MappedArray`, `BatchMixin`
- ✅ `_utils.py` — `convert_to_categorical_in_place` only — reviewed; the dead/duplicate covariate-encoder
  builders (`get_covariates_encoders_from_dict`/`get_covariate_encoder`/`get_label_encoder`/
  `get_one_hot_encoder`/`get_functional_encoder`) + the `TargetCovariates*` aliases were deleted (they
  duplicated `_encoders.py` and were unreferenced).
- ⬜ `containers/_base.py` — `BaseData`  🗑 (thin marker after dead-method removal; spec-only collapse deferred)
- ⬜ `containers/_categorical.py` — `CategoricalData` (`from_pandas`, `extract_reps`, `category_realms`)
- ⬜ `schemas/` — `DataSchema`/`StrictDataSchema`, `StateDataSchema`, `ConditionDataSchema`, `CouplingDataSchema`, `CovariatesDataSchema`, `ResponseDataSchema`
- ⬜ `sim/_dummy_adata.py` — `get_dummy_adata` 🗑 **relook/likely-delete**: only the sim-package re-export
  references it; no live consumer remains (the scripts/tests that used it were deleted).

### `sc_flow/flow/` (velocity fields, paths, objectives, predict)
- ⬜ `_vf.py` — keep `MLPVelocity` as the thin runtime module. ✅ removed unused
  `VelocityFieldFn`/`get_vf_fn` and the one-subclass `BaseVelocityField`; `MLPVelocity` is now an ordinary
  `torch.nn.Module`. Move the remaining config and persistence adapters off the runtime class as the
  registry/bundle work lands.
- ⬜ `_config.py` — `MLPVelocityConfig` -> registered `VelocityConfig`; `SetEncoderConfig` -> registered
  component config; 🗑 `MLPEmbedderConfig` and its helpers are replaced by generic `net` specs.
- ⬜ `_set_encoder.py` — `SetEncoder`
- ✅ `_pooling.py` — `BasePooling`, `MeanPooling`, `SumPooling`, `TokenAttentionPooling`, `SeedAttentionPooling`
  — reviewed; each config now **owns its `build(self, context)`** and registers via
  `@POOLING_REGISTRY.register("id")` (config-owns-build — no separate `_factory` fns); `build_pooling` /
  `validate_pooling_spec` façades kept. (The attention-port *numerics* were not re-audited here.)
- ✅ `_combiner.py` — `BaseCombiner`, `ConcatCombiner`, `Resnet1dCombiner` — reviewed; same config-owns-build
  pattern via `@COMBINER_REGISTRY.register`; `build_combiner` / `validate_combiner_spec` façades kept.
  The ResNet combiner now carries a canonical nested `scfit.resnet` spec instead of
  `num_resnet_layers` + an unchecked `resnet_kwargs` forwarding bag.
- ⬜ `_objectives.py` — `LinearFMObjective`, `OTFMObjective`, `GENOTObjective`
- ✅ `_predict.py` — `ODEPredictor` (`predict`, `predict_with_aux` → trajectory aux), `integrate_translation`,
  `condition_to_device`, `condition_mask_to_device` — reviewed; no longer `@register_predictor` (registry
  dropped). It is the single inference engine — instantiated directly and shared by validation and
  `FlowMatching.predict`.
- ✅ `_validation.py` — `PerturbationValidationCallback` — new; the control→perturbed population validation
  (predict via an injected `Predictor`, score model + identity baseline, optional source cap) as a
  Lightning `Callback`, so the generic `scfit` harness stays perturbation-free. Inert without a
  predictor + metrics (validation is optional).
- ⬜ `_time_features.py` — `sinusoidal_time_features`, `log_sinusoidal_time_features`, `get_time_features_fn`
- ⬜ `_torch_types.py` — generic torch aliases (relocated from core)
- ⬜ `_torch_utils.py` — `to_torch_tensor`, `broadcast_to_target_shape`, `make_concatenation_possible`, `ensure_2d_tensor_with_singleton_trailing_dim`, `get_torch_device`
- ⬜ `probability_paths/_probability_paths.py` — `BaseProbabilityPath` + 5 concrete paths
- ⬜ `coupling/` — `ot_linear_coupling`, `ot_quadratic_coupling`, `independent_coupling`, DLPack `couple_device`, `OTFn`/`OTResult`, jax utils
  — ⚠️ `independent_coupling` is exported in `coupling.__all__` but has **no consumer**: the objective's
  `match_method == "independent"` path (`_objectives.py:151`) doesn't call it. **relook** — dead export or missing wiring.

### `sc_flow/` (facade)
- ⬜ `_model.py` — `FlowMatching` (`fit`, `predict`, `save`, `load`) — *partially reviewed*: `predict` unified
  onto `ODEPredictor` (adds `return_aux` → `(pred, {"trajectory": ...})`, replacing `return_trajectory`);
  `fit` builds the generic `TrainingModule` and attaches `PerturbationValidationCallback` (validation
  optional). `save`/`load` and the objective wiring not yet reviewed.
- ⬜ `_types.py` — `ProbabilityPathId`, `TimeFeaturesId` (flow-only; generic types now in `scfit._types`)
- ⬜ `_constants.py`
- ⬜ `_optional.py` — `require`

---

## Follow-ups flagged during consolidation
- ✅ **`_utils` deduplicated** — `scfit._utils` is the single owner; `sc_flow/_utils.py` deleted, `sc_flow` imports from `scfit._utils`.
- 🗑 **generic `_types` can soon disappear** — the perturbation-specific `TargetCovariates*` aliases and
  unused `NestedLayersDict` are removed; `LayersDict` remains coupled only to the transitional
  `init_module_from_dict`. Delete both with the `net`-spec rewrite.
- ✅ **`CategoricalData.concat_collection`** removed (+ `__getitem__`/`__len__`); **`BaseData`** reduced to a marker in `sc_flow/data/containers`. The spec-only collapse (drop `ann_df`, fold `BaseData` away) remains the deferred deeper refactor.
- ✅ **training module genericized** — `scfit.training.TrainingModule` is training-only (+ `prog_bar_metrics`); the perturbation held-out scoring lives in `sc_flow.flow.PerturbationValidationCallback`. A generic base trainer no longer knows a batch has `source`/`target`.
- ✅ **predictor registry dropped** — `Predictor` ABC kept (+ `predict_with_aux`); `register_predictor`/`build_predictor`/`PREDICTOR_REGISTRY` deleted; `ODEPredictor` instantiated directly by the facade.
- ✅ **inference unified** — `FlowMatching.predict` routes through the same `ODEPredictor` as validation (single inference path); `return_aux` surfaces the ODE trajectory (was `return_trajectory`).
- ✅ **config-owns-build** — `ComponentRegistry.register` is dual-mode; pooling & combiner configs each own `build(self, context)` and register via decorator (no separate factory functions).
- 🗑 **`init_module_from_dict` is now superseded** — dropping its dead one-option `layer_type` branch was
  a useful intermediate cleanup, but the `net` family removes the helper and `_resolve_dim` entirely.
- ✅ **dead covariate-encoder path removed** — `sc_flow/data/_utils.py` reduced to `convert_to_categorical_in_place`; the `get_*_encoder*` builders + `TargetCovariates*` aliases (duplicated `_encoders.py`) deleted.
- ✅ **G1–G3 base-module cleanup landed** — `BaseModule` and `_make_modules` are gone; construction belongs
  to configs or local concrete-module helpers, and only the top-level bundle will own portable export.
  This also removes the `huggingface-hub` dependency, constructor sniffing, runtime-module repair
  attributes, and the one-subclass `BaseVelocityField`.
- ✅ **typed `scfit.resnet` landed** — the first explicitly approved G4 slice introduces the generic
  `NET_REGISTRY`/`NetContext` surface and replaces `Resnet1dCombiner.resnet_kwargs`. The authored spec
  contains only architecture hyperparameters (`num_layers`, activation/norm/dropout/bias); the combiner
  context supplies tensor widths. This removes the last signature-reflection utility.
- ✅ **condition contract flipped — data-side lookup, model-side encoding.** The dataloader emits per
  realm an **integer index** (categorical) or a **feature vector** (lookup); learnable encoding lives
  model-side in the `REALM_ENCODER_REGISTRY` discriminated-spec family
  (`sc_flow.embedding`/`onehot`/`feature_mlp`, config-owns-build, in
  `flow/_realm_encoders.py`). `compile_obs` produces a pure `condition_lookup`;
  `CompiledDims` carries `condition_num_categories`; `SetEncoderConfig.input_layers` became `realms`;
  `_build_vf` builds the realm specs; and `FlowMatching` stores `_condition_lookup`.
  `scfit.data.Loader` and `EvalLoader` now consume the same `ConditionLookup` contract: a leaf maps to
  a non-empty `{realm: np.ndarray}` whose arrays have a leading singleton batch axis. The unpublished
  `condition_fn` name and the legacy unstructured-array path were removed directly. The data layer
  validates the mapping, shape, and numeric-kind contract and **preserves each array and its dtype
  unchanged**. Dtype/device conversion happens only at the compute boundaries: the objective and
  predictor move values to the model device, convert floating features to the model floating dtype,
  and convert integer indices to `torch.long` for embedding/one-hot consumers. The
  `OneHot`/`IndexEmbedding` modules do not coerce their input indices. Covariate migration, the
  per-realm onehot/embedding facade knob (`condition_encoding`),
  `categorical()` data-side declaration, and realm encoder/config round-trips are done and covered by
  the pipeline and realm-encoder smokes. **Only open:** the additive LLM/text realm encoder, deferred
  by request.
- ⬜ **objective registry → `ComponentRegistry`** — deferred; migrate with the predictor pattern when portable training-config / entry-point plugins are actually needed.
- ⬜ **component `version` = informational stamp, not a migration guarantee** — decision, checked against the ecosystem: the norm is a *single global* version stamp (HF `transformers_version`, diffusers `_diffusers_version`) that is informational/telemetry, **not** per-component migration — HF handles old checkpoints in code (defaults absorb renamed fields), not by branching on the number; timm carries **no** schema version (name + lib version); only **ONNX** does true per-unit versioning, and it backs it with *retained old operator versions + resolution rules* — machinery we don't have (our `_parse` only **rejects** anything `!= 1`). Versioning granularity should match *who evolves independently*: one library + built-ins migrated together → one global stamp (us, today); many independent producers on separate schedules (ONNX operators, third-party plugins) → per-unit. So:
  - **(a) The bundle `FORMAT_VERSION` is the real compat mechanism** and is already achievable — it exists (`flow/_config.py:39`) and is already *written* to `config.json` (`_model.py:490`). The only gap: `load` reads the file but never checks it. Finish it by **gating on `format_version` at load** (warn/raise on mismatch). Small change, no new machinery.
  - **(b) Keep the per-spec `version` field as a cheap reserved slot** (the wire format is expensive to change; reserving one integer now lets us adopt ONNX-style per-type migration later without a breaking change) — but describe it honestly as *informational + rejects-unknown*, not a migration guarantee. Build actual `migrate(old)->new` per type only when independent evolution is real: the first third-party plugin, or the first backward-incompatible change to a *shipped* component.
  - **(c) Default `version=1` in one place and delete the duplication.** `ComponentRegistry.register(..., versions=(1,))` already defaults it, so registration never repeats it. Remove the hand-rolled `ARCHITECTURE_VERSION` + `MLPVelocityConfig.from_spec` version check (`flow/_config.py:488-497`) — it re-implements the envelope + version validation `ComponentRegistry.validate`/`_parse` already do; route the architecture envelope through the shared machinery (single-member registry, or a shared `to_spec`/`from_spec` base defaulting `version=1`). One place owns "version defaults to 1, validated here."

### ⬜ DESIGN HANDOFF: one declarative experiment, staged from the model tree outward

**Status: design, except for explicitly approved slices.** This section defines the destination and the
boundary of the next rewrite. G1–G3 and the typed `scfit.resnet` slice of G4 are implemented; the
remaining G4/model/data/pipeline work stays at design stage.

#### Long-term contract

A pipeline author edits one declarative experiment config, not Python. The resolved config describes the
data contract, architecture, objective, fit policy, validation, and export. The same config trains,
evaluates, predicts, and exports; changing a component or scaling a run is a config edit. YAML is the
authoring format, while canonical JSON is the persistence/wire format.

Every polymorphic value is a versioned discriminated spec:

```yaml
type: sc_flow.velocity
version: 1
config: {...}
```

The full experiment must satisfy:

```python
canonical = ExperimentConfig.from_config(raw).to_config()
assert json.loads(json.dumps(canonical)) == canonical
```

Construction is one directional pipeline:

```text
resolved experiment config
  -> resolve_source(data.source)
  -> FlowSpec.from_config(data.flow)
  -> FlowSpec.compile(source) -> fitted input schema + stream plan
  -> ARCHITECTURE_REGISTRY.build(architecture, ArchitectureContext(input_schema))
  -> OBJECTIVE_REGISTRY.build(objective, ObjectiveContext(...))
  -> fit(model, stream, FitConfig, ValidationConfig)
  -> public prediction from the same predictor spec used by validation
  -> portable bundle
```

The model tree is **data-shape independent**. State width, lookup width, category counts, and other
observed dimensions come from the build context produced by the fitted input schema; they are not copied
into the authored architecture config. This is what permits the same reviewed architecture spec to train
against compatible datasets and what lets load reconstruct the model without reopening the training data.

An illustrative final config (field names under the deferred sections are not frozen yet):

```yaml
format_version: 1
run: {name: tahoe_otfm, seed: 42}

data:
  source: {uri: tahoe100_grouped/plate*.zarr}
  flow:
    state: {sample_rep: X_pca}
    control_key: is_control
    conditions:
      drug:
        columns: [drug]
        encoder: {type: sc_flow.categorical, version: 1, config: {}}
    match_context: [cell_line]

architecture:
  type: sc_flow.velocity
  version: 1
  config:
    state_latent_dim: 256
    time_latent_dim: 64
    state_encoder:
      {type: scfit.mlp, version: 1, config: {hidden_dims: [2048, 2048]}}
    time_encoder:
      {type: scfit.mlp, version: 1, config: {hidden_dims: [256]}}
    condition_encoder:
      type: sc_flow.set_encoder
      version: 1
      config:
        realms:
          drug: {type: sc_flow.embedding, version: 1, config: {output_dim: 256}}
        pooling: {type: sc_flow.mean, version: 1, config: {}}
        output_dim: 256
    combiner: {type: sc_flow.concat, version: 1, config: {}}
    decoder:
      {type: scfit.mlp, version: 1, config: {hidden_dims: [4096, 4096]}}

objective:
  type: sc_flow.otfm
  version: 1
  config:
    probability_path: {type: sc_flow.linear_dirac, version: 1, config: {sigma: 0.0}}
    match: {type: sc_flow.sinkhorn, version: 1, config: {}}

fit:
  sampler: {batch_size: 1024, chunk_size: 32, preload_nchunks: 4096}
  optimizer: {lr: 0.0001}
  trainer: {max_steps: 30000, accelerator: gpu}
  split: {by: [drug], ratios: {train: 0.8, val: 0.1, test: 0.1}}

validation:
  predictor: {type: sc_flow.ode, version: 1, config: {num_steps: 50}}
  metrics: [mean_aggregated_r_squared, e-dist]
```

Layered YAML is still allowed as an authoring convenience (`base` + experiment + machine/profile
override). A run records the **one fully resolved canonical config** plus the pipeline commit and lockfile;
the layers themselves are not the reproducibility contract.

#### Component contract

`ComponentSpec` remains the `{type, version, config}` JSON envelope. A registered config dataclass owns
validation of its own scalar fields and `build(context)`. `ComponentRegistry` keeps only:

- `validate(spec) -> canonical ComponentSpec`
- `parse(spec) -> registered config dataclass`
- `build(spec, context) -> runtime object`

There are no per-family `build_*`/`validate_*` facades, `_make_*` construction hooks, or hand-written
`to_dict`/`from_dict`/`to_spec`/`from_spec` serializers.

**Important correction to the earlier handoff:** nested specs must **not** be eagerly replaced by child
config dataclasses. `dataclasses.asdict()` would then discard every child `{type, version}` envelope and
break the JSON round trip. A parsed composite config keeps each nested slot as a canonical
`ComponentSpec`. Its `__post_init__` canonicalizes that slot through the correct family registry, and its
`build(context)` recursively calls that registry's `build`. This extends the pattern already used by
`SetEncoderConfig.pooling`/`realms`, preserves wire identity, and lets one top-level `validate` cascade
through the tree without magical reflection over arbitrary dictionaries.

Every config-only entry point accepts mappings (including resolved OmegaConf mappings) but stores plain
Python JSON values. Unknown fields fail loudly and nested errors name the full slot path. Runtime
`nn.Module` instances remain a programmatic research escape hatch: they may train, but are not accepted by
the portable config path and `save` fails before writing anything, naming the non-portable slot.

#### Registry and type ownership

Recommended final ownership (G4; the `scfit.resnet` slice is implemented, the remainder still requires
approval):

- `scfit` owns `ComponentSpec`/`ComponentRegistry`, the generic `net` family, and generic implementations
  such as `scfit.mlp` and `scfit.resnet`.
- `scfit` owns the generic `architecture` family surface; a concrete toolbox registers its own
  namespaced architectures.
- `sc_flow` owns `sc_flow.velocity`, set/realm encoders, pooling, combiners, probability paths,
  objectives, predictors, and their contexts.
- Concrete type IDs name the package that owns their semantics. They are persistent wire identities, not
  class names; renaming one later requires an explicit version migration.
- Entry-point discovery is deferred until a second external toolbox needs it. Direct package import is
  sufficient for the first implementation; do not design a plugin protocol speculatively.

The names to settle before configs proliferate:

- top-level `sc_flow.mlp_velocity` -> `sc_flow.velocity`
- generic nets -> `scfit.resnet` (landed) and `scfit.mlp` (pending)
- the current combiner `sc_flow.resnet1d` -> `sc_flow.resnet_fusion`
- the velocity trunk, stream embedders, feature-realm projection, and set-encoder output head become
  nested `net` specs; slot context supplies input/output dimensions

Categorical realm configs are dimension-free too: `sc_flow.embedding` carries its desired output width
but receives `num_categories` from the fitted input schema; `sc_flow.onehot` receives its full width from
that same context.

#### What happens to `MLPVelocity`

`MLPVelocity` does **not** become another config object and it does not disappear. It becomes the thin
runtime `torch.nn.Module` produced by the registered `sc_flow.velocity` config:

- `MLPVelocityConfig` becomes a registered `VelocityConfig` (wire type `sc_flow.velocity`). It owns the
  component specs, validates them, derives child build contexts from `ArchitectureContext(input_schema)`,
  and constructs the runtime module.
- `MLPVelocity` owns only tensor behavior: `forward`, `condition_stats`, and
  `velocity_from_embedding`. Its constructor receives the already-built child modules. It does not retain
  a second serializable config graph.
- `is_conditional` and source/stochastic behavior are inferred from the installed child modules (or
  explicit primitive flags), not delegated back to a stored config object.
- `to_config`/`from_config` and `save_pretrained`/`from_pretrained` leave the runtime class. Registry
  construction and the `FlowMatching` bundle assembler own those operations. Temporary adapters may stay
  only until the bundle TODO lands.
- Keep the Python class name during this rewrite to avoid an unrelated public-API rename. The stable
  identity is the wire type `sc_flow.velocity`; the runtime class can be renamed later without a config
  migration if the swappable child nets make `MLPVelocity` too narrow a name.

`VelocityFieldFn` and `get_vf_fn` had no live source consumer; both are now removed because the predictor
directly calls the module. `BaseVelocityField` was likewise only a one-subclass ABC and public re-export;
G3 removes it rather than preserving an incomplete extension contract. If another runtime implementation
later needs an interface, define a protocol from the methods the objective/predictor actually consume.

`SetEncoder` follows the same split: its registered config builds it; the runtime class keeps `forward`
and runtime properties actually consumed by the velocity field. Its unused `from_config`/`to_config`,
`pooling_spec`, `pooling_output_dim`, and `decoder_input_dim` pass-throughs are removed.

#### Scope of the next model-side rewrite

1. Replace the pooling/combiner/realm `build_*` and `validate_*` facades with registry verbs.
2. Register the top-level velocity, set encoder, generic nets, pooling, combiner, and realm encoders as
   config-owning component dataclasses. `scfit.resnet` is complete; `scfit.mlp` and the remaining
   top-level/composite registrations are pending.
3. Replace composite hand-written serializers and `ARCHITECTURE_TYPE` constants with registry validation.
4. Make every nested polymorphic model slot a canonical spec and recursively validate/build it.
5. Remove one-option MLP dictionaries in favor of the `net` family; keep fixed algorithmic enums as
   strings where they are genuinely closed choices.
6. ✅ Remove the dead `_make_modules` stubs and inline live construction. Completed with G1–G3.
7. Keep `FlowMatchingConfig` as a temporary high-level compatibility recipe. It may materialize the new
   architecture spec internally during this rewrite, but the final `ExperimentConfig` will carry the
   architecture tree directly. Until the bundle TODO lands, the legacy save format may still contain its
   derived architecture graph, but it is a cache of the recipe, never an independently editable source.
8. Delete the remaining replacement-coupled helpers: `LayersDict`, `init_module_from_dict`,
   `FunctionalModule`, and the `MLPVelocity` runtime config/persistence adapters listed above.
   `BaseVelocityField`, the immediately orphaned signature utilities, `activation_id`,
   `NestedLayersDict`, `VelocityFieldFn`, and `get_vf_fn` are already removed.

#### Gate status

The owner explicitly approved G1–G3; they landed as one sequence so there is no half-migrated
serialization contract:

- ✅ **G1 — abstract `_make_modules` removed.** Config-owned construction and concrete local build helpers
  replace the artificial subclass hook.
- ✅ **G2 — `BaseModule.__new__` constructor sniffing and later `object.__setattr__` repairs removed.**
  Portability is a property of the config tree and will be checked once by the bundle assembler.
- ✅ **G3 — `BaseModule` dissolved.** Leaves are ordinary `torch.nn.Module`s; the temporary
  `MLPVelocity.save_pretrained` adapter remains only until `FlowMatching` becomes the bundle boundary.

G4 is now partially authorized:

- 🟨 **G4 — generic `net`/`architecture` registries in `scfit`.** The generic `net` surface and
  `scfit.resnet` are approved and landed for the ResNet-combiner migration. `scfit.mlp`, the generic
  `architecture` family, and broader component-tree conversion remain gated. Entry-point discovery
  remains a later addition.

#### Explicit TODOs — required for the vision, intentionally outside this model-side rewrite

1. **Data config and fitted schema.** Add `FlowSpec.from_config`/`to_config`; replace
   `build_encoder("lookup:<key>")` with a `data_encoder` component family. `functional(fn, inv)` remains
   runtime-only and makes export fail. Define `input_schema.json` with ordered realms, representation
   widths, category vocabularies, `max_comb`, coupling locations, and asset references. Numeric fitted
   lookup tables live in `assets.safetensors`; JSON metadata stays in the schema. `FlowSpec.build_loader`
   remains a data-stream operation, not a model component.
2. **Fit and validation config.** Replace the long `FlowMatching.fit(...)` keyword surface with a strict,
   JSON-round-trippable `FitConfig` (sampler, split, optimizer, trainer) plus a strict
   `ValidationConfig`. Remove callback objects from the portable path or represent supported callbacks as
   specs.
3. **Objective and predictor specs.** Replace `objective="otfm"` +
   `build_objective(name, ...)` and direct `ODEPredictor(...)` construction with their own
   `ComponentRegistry` families. Probability path/matching policy should be nested objective specs;
   validation and public prediction must build the same predictor spec. Do not infer predictor behavior
   from an objective-name string.
4. **Portable bundle.** Make `FlowMatching.save/load` a symmetric bundle assembler:
   `manifest.json`, canonical `config.json`, `input_schema.json`, `model.safetensors`, and optional
   `assets.safetensors`. Loading reconstructs the Python object graph from JSON specs and uses tensor files
   only for state. Delete `data_state.pkl`/cloudpickle and stop persisting both a recipe and a derived
   architecture graph.
5. **Top-level experiment config.** Introduce the public object that composes data, architecture,
   objective, `FitConfig`, predictor/validation, and export. This supersedes the temporary
   `FlowMatchingConfig` shorthand.
6. **`cf-train` migration.** Keep its submitit/Optuna/profile/path orchestration, but delete the duplicated
   model-facing `DataCfg`/`ModelCfg`/`TrainerCfg` schema and the manual
   `prepare_data -> prepare_model -> train` glue. Its training entry point should resolve paths, hand the
   experiment mapping to the library-owned strict config parser, run it, and record the resolved config +
   commit + lockfile. Migrate `configs/base.yaml` and `configs/experiments/*` only after the public
   experiment schema is stable. Remove the `cellflow-tools`/old `dagloader` dependency path rather than
   adapting that legacy API; pre-flight checks should delegate schema/data validation to `FlowSpec`.
7. **Optional extensions.** LLM/text realm encoder, metric specs, external component entry points, and
   generic multi-toolbox dispatch. None blocks the first complete `sc_flow` pipeline.

Dependency order: model component tree -> data config/input schema -> `FitConfig` + typed
objective/predictor -> bundle -> top-level experiment config -> `cf-train`.

#### Gates

For the model-side rewrite:

- `smoke_pipeline.py` and `smoke_realm_encoders.py`
- top-level conditional-velocity `validate -> JSON -> parse -> build` round trip
- unknown/missing nested fields fail with a slot-qualified message
- dimension-free architecture builds from two compatible input-schema contexts
- runtime-module escape hatch trains, while portable save rejects it before creating files
- the existing save/load smoke remains green until the bundle TODO deliberately replaces it

For the final pipeline:

- a committed `cf-train` YAML runs train + held-out validation + export with no model-specific Python
- loading the bundle does not import cloudpickle or reopen the training source
- predictions before save and after load match under the same predictor spec
- the bundle contains the canonical resolved config and no `.pkl` files

#### Current implementation notes and remaining ledger items

- ✅ **`FlowMatching` recipe is a config object** — `FlowMatchingConfig` (kw-only dataclass, in `sc_flow._model`, **out of the `scfit` core**) replaces the long kwargs list; construct via `FlowMatching(spec, config)` or `FlowMatching.from_config(spec, mapping_or_OmegaConf)`. `save` persists `self.config.to_dict()` (single recipe object; dropped the ad-hoc `_CTOR_FIELDS`, and `condition_encoding` is now actually persisted — it was silently missing before), `load` rebuilds via `FlowMatchingConfig.from_dict`. Verified: OmegaConf `from_config` pipeline + save/load round-trip.
- ⬜ **`FlowMatching` save/load → bundle assembler** — remaining slim-down: `config.json` still stores both the `flow_matching` recipe *and* the realized `architecture` graph (both aid reconstruction, but the graph is derivable from recipe+dims). Reduce `save`/`load` to assembling `VF.save_pretrained` (config.json + safetensors) + `input_schema.json` + `assets.safetensors` + `manifest.json`, deleting `data_state.pkl` (the last cloudpickle). Gated on the biological-schema work (`input_schema.json` replacing the pickled spec/dims/`condition_lookup`).
- ✅ **smoke gate** — `scripts/smoke_harness_split.py` covers the trainer/validation split, optional-validation, the `PerturbationValidationCallback` (model + identity streams), and `Predictor.predict_with_aux` (no data pipeline / jax needed).
- Docstrings on all ⬜ items are being stripped (pending step) so review starts from the code, not stale prose; ✅ binded docstrings are kept.
- `scfit/data` naming/dead-code tidy-up is running as a separate task (`_resolve_source` → `_resolve_container`, etc.).
- ⬜ **binded: clearer error on empty `match_context`** — an empty match-context leaves the control `Node` with zero grouping columns, surfacing as a cryptic pandas `ValueError: Must pass non-zero number of levels/codes` (`node 'ctrl': ...`). binded should validate a non-empty bind `common` (or a genuinely single-leaf control) up front with an actionable message. Found via `scripts/smoke_pipeline.py`. *(binded / scfit.data.)*
- **Defined-but-unused-in-own-file sweep** (module-level functions; `_resolve_config_map`-style). Rule applied:
  used-elsewhere/exported → *rename/relook*, unused-anywhere & unexported → *delete*.
  - `scfit/data/_schema.py` — `_resolve_config_map`, `_weight_vector`: private but imported by
    `_loader`/`_split`/`_backend`/`_io`. **relook/rename** (de-facto internal API marked `_`). *(binded — folds
    into the tidy-up task above.)*
  - `scfit/data/_backend.py` — `_resolve_source`, `_node_stats`, `_bind_on`, `_build_loaders`: same
    private-cross-module pattern (used by `_loader`/`_eval_loader`). **relook/rename.** *(binded.)*
  - `sc_flow/data/sim/_dummy_adata.py` `get_dummy_adata` and `sc_flow/flow/coupling` `independent_coupling`
    — see their entries above (exported, no live consumer → relook).
  - Note: the sweep flagged `sc_flow/__init__.py:__getattr__` as unused — **false positive**, it's the PEP-562
    lazy-import hook the interpreter calls; **keep.** (Methods weren't statically swept — dynamic dispatch
    makes "unused method" unprovable by grep; this pass covers module-level functions only.)
