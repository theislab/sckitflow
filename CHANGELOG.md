# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [Unreleased]

### Added

- `predict_adata` names the source cell behind each prediction in `obs["source_obs_name"]` when the batch has a source (one name per row, aligned as the source state is, repeated per sample under `aggregate="samples"`); `EvalLoader.source_obs_names` holds them per group
- `predict_adata(batch_size=...)` caps the observations per `predict` call, splitting a larger group into batches whose predictions are concatenated back (replaces `chunk_size`; `progress_total` is removed, the bar counts groups again without reading the data twice). A split batch keeps its group's encoding and has its own `n_obs`, so predicting from noise without a source no longer fails. Requires scfit 0.0.12
- `EnergyDistance(max_samples=...)` and `MaximumMeanDiscrepancy(max_samples=...)` score a fixed random subset of at most that many rows per update: validation scores every sample, and the cost is quadratic in the rows

### Changed

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
- `VariancePreservingDiracProbabilityPath` ran from target to source: its mean was `cos(πt/2)·x1 + sin(πt/2)·x0`. It now runs from `x0` at `t=0` to `x1` at `t=1`, with the velocity to match
- With `n_samples` > 1, validation scored the mean over each row's samples, a distribution with 1 / `n_samples` of the predicted variance. It now scores every sample
