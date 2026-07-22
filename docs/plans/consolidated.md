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
- ⬜ class `ComponentSpec`
- ⬜ class `ComponentRegistry` (`register`, `validate`, `parse`, `build`, `family`, `types`)

### `scfit/_types.py`
- ⬜ `LayersDict`, `NestedLayersDict`, `TargetCovariatesEncodingId`, `TargetCovariatesEncoderCls`

### `scfit/_utils.py`  (single owner; `sc_flow` imports from here)
- ⬜ `check_type_against_generic`, `get_fn_args_names_and_types`, `get_fn_kwargs_names_and_types`,
  `verify_fn_args`, `verify_fn_kwargs_dictionary`, `verify_fn_signature`, `check_sequence_query_against_reference`

### `scfit/nn/`
- ⬜ `_modules.py` — `BaseModule` (`save_pretrained`), `FunctionalModule`, `MLP`, `Resnet1d`
- ⬜ `_activation.py` — `resolve_activation`, `activation_id`
- ⬜ `_utils.py` — `init_module_from_dict`

### `scfit/metrics/_metrics.py`
- ⬜ class `RSquared`, `EnergyDistance`, `MaximumMeanDiscrepancy`; `rbf_kernel_torch`

### `scfit/training/`
- ⬜ `_harness.py` — `TrainingModule` (`training_step`, `validation_step`, `on_validation_epoch_end`, `configure_optimizers`)
- ⬜ `_objective.py` — `Objective` (`compute_loss`), `register_objective`, `build_objective`, `OBJECTIVE_REGISTRY`
- ⬜ `_predictor.py` — `Predictor` (`predict`), `register_predictor`, `build_predictor`, `PREDICTOR_REGISTRY`

---

## ⬜ Pending review — `sc_flow` (flow-matching toolbox)

### `sc_flow/data/` (perturbation data layer — compiles onto `scfit.data`)
- ⬜ `_spec.py` — `FlowSpec` (`compile`, `build_loader`)
- ⬜ `_compile_obs.py` — `CompiledDims`, `CompiledData`, `compile_obs`
- ⬜ `_encoders.py` — `Encoder`, `OneHot`, `Label`, `Functional`, `Lookup`; `one_hot`, `label`, `lookup`, `functional`
- ⬜ `_abc.py` — `Distribution`, `MatchedDistributions`, `DataTree`
- ⬜ `_mixins.py` — `MappedTree`, `MappedArray`, `BatchMixin`
- ⬜ `_utils.py` — covariate-encoder builders, `convert_to_categorical_in_place`
- ⬜ `containers/_base.py` — `BaseData`  🗑 (thin marker after dead-method removal; spec-only collapse deferred)
- ⬜ `containers/_categorical.py` — `CategoricalData` (`from_pandas`, `extract_reps`, `category_realms`)
- ⬜ `schemas/` — `DataSchema`/`StrictDataSchema`, `StateDataSchema`, `ConditionDataSchema`, `CouplingDataSchema`, `CovariatesDataSchema`, `ResponseDataSchema`
- ⬜ `sim/_dummy_adata.py` — `get_dummy_adata`

### `sc_flow/flow/` (velocity fields, paths, objectives, predict)
- ⬜ `_vf.py` — `BaseVelocityField`, `MLPVelocity` (+ `to_config`/`from_config`/`save_pretrained`/`from_pretrained`)
- ⬜ `_config.py` — `MLPEmbedderConfig`, `SetEncoderConfig`, `MLPVelocityConfig`
- ⬜ `_set_encoder.py` — `SetEncoder`
- ⬜ `_pooling.py` — `BasePooling`, `MeanPooling`, `SumPooling`, `TokenAttentionPooling`, `SeedAttentionPooling` (+ configs, `build_pooling`, `validate_pooling_spec`)
- ⬜ `_combiner.py` — `BaseCombiner`, `ConcatCombiner`, `Resnet1dCombiner` (+ configs, `build_combiner`, `validate_combiner_spec`)
- ⬜ `_objectives.py` — `LinearFMObjective`, `OTFMObjective`, `GENOTObjective`
- ⬜ `_predict.py` — `ODEPredictor`, `integrate_translation`, `condition_to_device`, `condition_mask_to_device`
- ⬜ `_time_features.py` — `sinusoidal_time_features`, `log_sinusoidal_time_features`, `get_time_features_fn`
- ⬜ `_torch_types.py` — generic torch aliases (relocated from core)
- ⬜ `_torch_utils.py` — `to_torch_tensor`, `broadcast_to_target_shape`, `make_concatenation_possible`, `ensure_2d_tensor_with_singleton_trailing_dim`, `get_torch_device`
- ⬜ `probability_paths/_probability_paths.py` — `BaseProbabilityPath` + 5 concrete paths
- ⬜ `coupling/` — `ot_linear_coupling`, `ot_quadratic_coupling`, `independent_coupling`, DLPack `couple_device`, `OTFn`/`OTResult`, jax utils

### `sc_flow/` (facade)
- ⬜ `_model.py` — `FlowMatching` (`fit`, `predict`, `save`, `load`)
- ⬜ `_types.py` — `ProbabilityPathId`, `TimeFeaturesId` (flow-only; generic types now in `scfit._types`)
- ⬜ `_constants.py`
- ⬜ `_optional.py` — `require`

---

## Follow-ups flagged during consolidation
- ✅ **`_utils` deduplicated** — `scfit._utils` is the single owner; `sc_flow/_utils.py` deleted, `sc_flow` imports from `scfit._utils`.
- ✅ **generic `_types` deduplicated** — `scfit._types` owns `LayersDict`/`NestedLayersDict`/`TargetCovariates*`; `sc_flow/_types.py` keeps only flow-specific `ProbabilityPathId`/`TimeFeaturesId` and imports the rest from `scfit._types`.
- ✅ **`CategoricalData.concat_collection`** removed (+ `__getitem__`/`__len__`); **`BaseData`** reduced to a marker in `sc_flow/data/containers`. The spec-only collapse (drop `ann_df`, fold `BaseData` away) remains the deferred deeper refactor.
- Docstrings on all ⬜ items are being stripped (pending step) so review starts from the code, not stale prose; ✅ binded docstrings are kept.
- `scfit/data` naming/dead-code tidy-up is running as a separate task (`_resolve_source` → `_resolve_container`, etc.).
