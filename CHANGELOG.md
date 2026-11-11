# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [Unreleased]

### Added

- `DataManager.get_required_fields()`: the `anndata.acc` references (state, coupling and continuous reps, grouping and split columns) and `uns` tables the schema reads, so a caller can open only those parts of a file
- `AdaLNZeroConditioning` (`nn.conditioning.adaln_zero`): adaLN-Zero as in DiT, the state layer-normed without a learned affine and modulated by `(1 + gamma)`, `beta` projected from the SiLU of time and condition, the projection starting at zero as in FiLM
- `predict_adata` names the source cell behind each prediction in `obs["source_obs_name"]` when the batch has a source (one name per row, aligned as the source state is, repeated per sample under `aggregate="samples"`); `EvalLoader.source_obs_names` holds them per group
- `predict_adata(batch_size=...)` caps the observations per `predict` call, splitting a larger group into batches whose predictions are concatenated back (replaces `chunk_size`; `progress_total` is removed, the bar counts groups again without reading the data twice). A split batch keeps its group's encoding and has its own `n_obs`, so predicting from noise without a source no longer fails. Requires scfit 0.0.12
- Layers can be reps: the state (`target_sample_rep`, `source_sample_rep`), the coupling reps and the continuous covariates may each be `A.layers[k]`
- `EnergyDistance(max_samples=...)` and `MaximumMeanDiscrepancy(max_samples=...)` score a fixed random subset of at most that many rows per update: validation scores every sample, and the cost is quadratic in the rows

### Changed

- Breaking: every field of `DataManagerConfig` naming an array holds an `anndata.acc` reference (`A.X`, `A.layers[k]`, `A.obsm[k]`), written in a config as anndata's dotted string, `"X"`, `"layers.<k>"` or `"obsm.<k>"`, and saved back as that string. This covers `target_sample_rep`, `source_sample_rep`, `rep_lin`, `source_rep_quad`, `target_rep_quad`, `continuous_conditions` and `target_continuous_covs`. Bare names no longer work: `target_sample_rep: X_pca` is rejected with an error asking for `obsm.X_pca`, so existing configs and saved specs must name the field. The continuous covariates take `obsm.<k>` or `layers.<k>` (never `X`) and are still keyed by `k` in a batch and in the condition encoder's `continuous_input_layers`
- Breaking: nothing defaults to `.X`. `target_sample_rep` is required (`"X"` for `.X`), and `DataManager()` without a config is removed. A linear coupling term is streamed only when `rep_lin` names one: an unset `rep_lin` no longer means `.X`, and a matched training method whose coupling reads a rep the config does not name raises when it is built instead of matching on `.X`. Independent coupling reads only row counts, so it needs no coupling rep. `source_sample_rep: None` still means the sources use the targets' state rep
- Breaking for Python callers: the loaders (`Loader`, `EvalLoader`) and the schemas (`StateDataSchema`, `CouplingDataSchema`, `ConditionDataSchema`, `ResponseDataSchema`) take references instead of `obsm` keys, `target_sample_rep` is required by the loaders, and `sckitflow.data._loader.get_rep` and `_get_loc` are removed (`adata[ref]` and `ref.isin(adata)` replace them). Error messages print references, e.g. `A.obsm['X_pca']`. `get_required_fields()` returns the references in the order the schema declares them
- Breaking: `use_linear` is removed. Both quadratic reps without `rep_lin` is pure Gromov-Wasserstein; with `rep_lin` set, the linear term is used. One rep may now serve several coupling terms (e.g. `rep_lin` and `source_rep_quad`), which used to drop the linear term
- Rep keys must be non-empty and hold no `.`, since a saved config writes the key after a dot and would read `obsm.X_pca.v2` back as something else
- Requires scfit's `anndata.acc` reference API (selmanozleyen/scfit#1), pinned to its `feat/open-adata` branch through a uv source until it is released
- The velocity field is rebuilt from nested configs: `MLPVelocity` / `MLPVelocityConfig` (`velocity_field.mlp`) are replaced by `NeuralVelocity` / `NeuralVelocityConfig` (`nn.models.velocity`), whose `state_encoder`, `time_features`, `time_encoder`, `condition_encoder`, `source_encoder`, `conditioning` and `field_decoder` are configs of their own instead of flat `*_kwargs` and `*_id` arguments. The blocks move to `sckitflow.core.nn.blocks` and `sckitflow.core.nn.composite` with `nn.*` type ids (`mlp` is `nn.mlp`), and `Resnet1d`, `BaseVelocityField`, `get_time_features_fn`, `make_custom_time_features`, `ott_jax_time_features` and `torch_cfm_time_features` are removed. Specs and models saved before do not load; `docs/configs/cellflow.yaml` shows the new form
- `SetEncoder` is built from a `SetEncoderConfig` and the per-covariate input widths (`categorical_dims`, `continuous_dims`), which come from the data
- `FiLMConditioning`: `gamma` and `beta` are a linear map of the SiLU of time and condition, no longer passed through a SiLU themselves (that bounded both below about -0.28), and the state becomes `(1 + gamma) * x + beta`, so `gamma` is an offset from 1 as in the FiLM paper. The projection starts at zero, as DiT's, so the layer starts as the identity
- The `gpu` extra is replaced by `cupy-cuda12` and `cupy-cuda13` (scfit's extras of the same name, cupy with its CUDA toolkit wheels). The uv selectors pair torch and cupy by CUDA major: `cu126` takes torch from the cu126 index with `cupy-cuda12`, `cu130` with `cupy-cuda13`. Requires scfit 0.0.13
- Matching reports `coupling_fallback` with the training loss: 1 when the plan was non-finite or all zero and the batch was matched at random (this replaces the warning, which needed a host sync every step). The loss metric is a tensor, so neither is read back per step; plan sampling draws by inverse CDF, as ott-jax's `sample_joint`, with no limit on the plan's size
- Matching stays in torch on the batch's device: the couplings pass torch tensors to POT, sample pairs by inverse CDF over the plan and return index tensors (and a torch plan with `return_matrix=True`), without syncing with the host in sckitflow's own code; a non-finite or all-zero plan falls back to random pairs silently, as warning would need that sync. They draw from the step's `torch.Generator`, so the `rng` argument of the couplings, `Matcher.match` and `compute_loss` is replaced by `generator` or removed, and `generators` returns only the torch generator. The (fused) Gromov-Wasserstein configs take `inner_num_iter_max` and `inner_stop_thr` for POT's inner Sinkhorn solves: its default `stopThr=1e-9` is never reached in float32, and `1e-4` cut a fused GW solve on a 256 x 256 batch on an A100 from 36.5 s to 0.5 s
- `DataManager.get_data_dimensionalities` sizes each representation from its shape in the files whose cells read it, and the condition and group encodings by encoding one group the way the loaders do, so the dimensions and the batches come from one code path. The in-memory containers that only fed the dimensions (`DistributionData`, `StateData`, `MixedTypeData`, `CouplingData`), `DataManager.get_distribution_data` and the schemas' `get_data` are removed
- `DataManagerConfig.condition_state_key` and `target_categorical_covs_dict` are removed: only the dimensions honoured them, never the loaders, so a model sized with them trained on differently shaped batches

- `DataManagerConfig.sample_rep` is renamed `target_sample_rep`, next to the new `source_sample_rep`; the loaders' `sample_rep` argument likewise
- The torch stack (`torch`, `lightning`, `pot`, `torchdiffeq`, `torchmetrics`, `torchsde`) is a core dependency: `import sckitflow` always needed it. The `torch` extra is kept, empty, so `sckitflow[torch]` still installs. `cloudpickle` moves to the test extra, and the unused `sckitflow._runtime` is removed

- `RunConfig` / `Run` / `save_run` / `load_run` are replaced by `ModelConfig` / `Model` (the data schema, module, training and inference methods) with `Model.save_pretrained` / `Model.from_pretrained` in the Hugging Face Hub layout of `scfit.pretrained` (`config.json` plus `module/model.safetensors`); `from_pretrained` also takes a Hub repo id. How a training run goes (optimizer, seeds, batch size, steps) is the experiment's: `FlowDataModuleConfig` and `OptimizerConfig` are removed. A model reads back under the package versions that saved it
- `FlowDataModule` holds its `adata` for its whole life: `attach`, `set_predict_data`, `predict_dataloader` and its `control_adata` are removed. `load_from_checkpoint(path, adata)` builds it with the data, and `predict_adata` builds its eval loader per call.
- Every `AnnData` argument of the data stack (`DataManager`, `FlowDataModule`, the loaders, `predict_adata`) also takes a list of them, read as one dataset; they must agree on dims and feature names, and the `.uns` tables the schema reads are merged across them (an entry in several must match). Requires scfit 0.0.11
- The split is no longer part of a run: `RunConfig.splitter` / `splitter_seed` and `DataManager(splitter=...)` are removed. Write the split column beforehand and declare it with `split_by`.
- `control_adata` is removed everywhere (`DataManager.get_dataloaders` / `get_eval_loader`, the loaders, `predict_adata`). Controls are drawn from the streamed data by the schema, and the loader holds just them in memory.
- `predict_adata` returns every sample by default: with `n_samples` > 1 there is one row per (input row, sample), its obs repeated and the sample's index in `obs["sample_idx"]`. `aggregate="mean"` gives the old output, the mean over samples in `.X` and every sample in `obsm["raw_samples"]`
- Probability paths are rebuilt around `BasePath` (mean `alpha_t x0 + beta_t x1`, standard deviation `sigma_t`), and the config base `ProbabilityPathConfig` is now `PathConfig`. `LinearDiracProbabilityPath`, `LinearGaussianProbabilityPath`, `SchrodingerBridgeProbabilityPath`, `VariancePreservingDiracProbabilityPath`, `BaseProbabilityPath` and their configs are removed, and so are their ids (`probability_path.linear_dirac`, `probability_path.linear_gaussian`, `probability_path.schrodinger_bridge`, `probability_path.variance_preserving_dirac`). The new paths are `LinearPath`, `TrigPath` and `EncDecPath`, with configs `LinearPathConfig` (`probability_path.linear`), `TrigPathConfig` (`probability_path.trig`) and `EncDecPathConfig` (`probability_path.enc_dec`). The linear and trig configs default `sigma` to 0 and `EncDecPathConfig` requires `sigma > 0`; `sigma` must be finite and non-negative, and a `sampler_cfg` is required when `sigma > 0`. The linear and trig paths use the standard deviation `sigma * sqrt(t(1 - t))`, as `SchrodingerBridgeProbabilityPath` did, and the enc/dec path uses the encoder/decoder schedule `cos^2(pi t)` with standard deviation `sigma * sin^2(pi t)` and needs `sigma > 0`. Specs saved with the old ids do not load. `LinearDiracProbabilityPath` is `LinearPath(sigma=0)`, `SchrodingerBridgeProbabilityPath` is `LinearPath` and `VariancePreservingDiracProbabilityPath` is `TrigPath(sigma=0)`; `LinearGaussianProbabilityPath`'s constant noise has no replacement, the closest is `LinearPath` with noise that vanishes at the endpoints

### Removed

- `sckitflow.preprocessing`, `sckitflow.external`, `sckitflow.data.splitters`, `docs/scripts/fit_cfm.py` and the `scripts` extra. Splitting, PCA and the training script live in [flowrun](https://github.com/selmanozleyen/flowrun) now

### Added

- `DataManagerConfig.source_sample_rep`: the sources may have their own state rep (e.g. ATAC controls flowing into RNA targets), of any width (`DataDimensions.source_state_dim`). The velocity field reads it through its source encoder, which is then required, and a flow method starting from the sources (`generate_from_noise=False`) refuses it; generating from noise conditioned on the encoded source, as GENOT does, learns the map
- A dataset split across files only needs each representation where the cells reading it are: the loaders route the files holding targets and those holding sources to separate scfit sources, so e.g. `source_rep_quad` can live only in the controls' file. A file missing a representation its cells need raises an error naming the file, the side and the schema field
- `sckitflow.core.surrogate.SurrogatePotential`: a differentiable potential of continuous condition covariates through a trained inference method, averaged over the groups of its context data (`adata`), which it reads once per device onto the inference module's device. It takes a `readout` and a `potential`; `MeanSamplesReadoutConfig` and `SquaredErrorPotentialConfig` build the common ones
- Basic tool, preprocessing and plotting functions
- `RunConfig.build` and `load_run` take `control_adata`, and `RunConfig.build` takes `metrics`


### Changed

- Flow method configs take only configs: live probability paths and raw sampler callables are rejected
- Time and noise samplers are separate families (`TimeSamplerConfig`, `NoiseSamplerConfig`)

### Fixed

- `MLP` no longer normalizes or drops out its output layer. With `use_layernorm` (no affine) every output row had mean 0 and variance 1, so a velocity decoder could not express a drift: a flow from noise to data away from zero (e.g. logicle-scale cytometry) stayed at the noise. Saved models with `use_layernorm`, `use_batchnorm` or dropout in an MLP load as before but predict differently
- Gromov-Wasserstein and fused Gromov-Wasserstein couplings default to `scale_cost="mean"`, like the linear ones: unscaled float32 costs made entropic GW underflow to an all-zero plan. A batch whose plan is all zero still falls back to random pairs; the training plan warns once per epoch with how many batches did
- Pure Gromov-Wasserstein matching (`use_linear=False`) no longer fails on the missing linear term ([#145](https://github.com/theislab/sckitflow/issues/145))
- OT couplings sample matched pairs from the plan with replacement. Drawing without replacement pulled pair frequencies toward uniform (off-diagonal pairs of `[[.45, .05], [.05, .45]]` came out ~35% too often)
- `TrainingPlan` keys its training generators by rank, optimizer step and batch index: with `accumulate_grad_batches > 1` every micro-batch of a step drew the same time, noise and matching randomness, and every DDP rank drew the same noise. `predict_adata` now keys its generators like `TrainingPlan.predict_step`, so both give the same predictions for a seed
- With `n_samples` > 1, validation scored the mean over each row's samples, a distribution with 1 / `n_samples` of the predicted variance. It now scores every sample
