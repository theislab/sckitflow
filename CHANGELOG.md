# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [Unreleased]

### Changed

- `DataManagerConfig.sample_rep` is renamed `target_sample_rep`, next to the new `source_sample_rep`; the loaders' `sample_rep` argument likewise
- The torch stack (`torch`, `lightning`, `pot`, `torchdiffeq`, `torchmetrics`, `torchsde`) is a core dependency: `import sckitflow` always needed it. The `torch` extra is kept, empty, so `sckitflow[torch]` still installs. `cloudpickle` moves to the test extra, and the unused `sckitflow._runtime` is removed

- `RunConfig` / `Run` / `save_run` / `load_run` are replaced by `ModelConfig` / `Model` (the data schema, module, training and inference methods) with `Model.save_pretrained` / `Model.from_pretrained` in the Hugging Face Hub layout of `scfit.pretrained` (`config.json` plus `module/model.safetensors`); `from_pretrained` also takes a Hub repo id. How a training run goes (optimizer, seeds, batch size, steps) is the experiment's: `FlowDataModuleConfig` and `OptimizerConfig` are removed. A model reads back under the package versions that saved it
- `FlowDataModule` holds its `adata` for its whole life: `attach`, `set_predict_data`, `predict_dataloader` and its `control_adata` are removed. `load_from_checkpoint(path, adata)` builds it with the data, and `predict_adata` builds its eval loader per call.
- Every `AnnData` argument of the data stack (`DataManager`, `FlowDataModule`, the loaders, `predict_adata`) also takes a list of them, read as one dataset; they must agree on dims and feature names, and the `.uns` tables the schema reads are merged across them (an entry in several must match). Requires scfit 0.0.11
- The split is no longer part of a run: `RunConfig.splitter` / `splitter_seed` and `DataManager(splitter=...)` are removed. Write the split column beforehand and declare it with `split_by`.
- `control_adata` is removed everywhere (`DataManager.get_dataloaders` / `get_eval_loader`, the loaders, `predict_adata`). Controls are drawn from the streamed data by the schema, and the loader holds just them in memory.

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

- Gromov-Wasserstein and fused Gromov-Wasserstein couplings default to `scale_cost="mean"`, like the linear ones: unscaled float32 costs made entropic GW underflow to an all-zero plan. A batch whose plan is all zero still falls back to random pairs, but now warns (`RuntimeWarning`) instead of only logging
- Pure Gromov-Wasserstein matching (`use_linear=False`) no longer fails on the missing linear term ([#145](https://github.com/theislab/sckitflow/issues/145))
- OT couplings sample matched pairs from the plan with replacement. Drawing without replacement pulled pair frequencies toward uniform (off-diagonal pairs of `[[.45, .05], [.05, .45]]` came out ~35% too often)
- `TrainingPlan` keys its training generators by rank, optimizer step and batch index: with `accumulate_grad_batches > 1` every micro-batch of a step drew the same time, noise and matching randomness, and every DDP rank drew the same noise. `predict_adata` now keys its generators like `TrainingPlan.predict_step`, so both give the same predictions for a seed
- `VariancePreservingDiracProbabilityPath` ran from target to source: its mean was `cos(πt/2)·x1 + sin(πt/2)·x0`. It now runs from `x0` at `t=0` to `x1` at `t=1`, with the velocity to match
