# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [Unreleased]

### Changed

- `RunConfig` / `Run` / `save_run` / `load_run` are replaced by `ModelConfig` / `Model` (the data schema, module, training and inference methods) with `Model.save_pretrained` / `Model.from_pretrained` in the Hugging Face Hub layout of `scfit.pretrained` (`config.json` plus `module/model.safetensors`); `from_pretrained` also takes a Hub repo id. How a training run goes (optimizer, seeds, batch size, steps) is the experiment's: `FlowDataModuleConfig` and `OptimizerConfig` are removed. A model reads back under the package versions that saved it
- `FlowDataModule` holds its `adata` for its whole life: `attach`, `set_predict_data`, `predict_dataloader` and its `control_adata` are removed. `load_from_checkpoint(path, adata)` builds it with the data, and `predict_adata` builds its eval loader per call.
- Every `AnnData` argument of the data stack (`DataManager`, `FlowDataModule`, the loaders, `predict_adata`) also takes a list of them, read as one dataset; they must agree on dims and feature names, and the `.uns` tables the schema reads are merged across them (an entry in several must match). Requires scfit 0.0.11
- The split is no longer part of a run: `RunConfig.splitter` / `splitter_seed` and `DataManager(splitter=...)` are removed. Write the split column beforehand (a `sckitflow.data.splitters` splitter does) and declare it with `split_by`.
- `control_adata` is removed everywhere (`DataManager.get_dataloaders` / `get_eval_loader`, the loaders, `predict_adata`). Controls are drawn from the streamed data by the schema, and the loader holds just them in memory.

### Added

- A dataset split across files only needs each representation where the cells reading it are: the loaders route the files holding targets and those holding sources to separate scfit sources, so e.g. `source_rep_quad` can live only in the controls' file. A file missing a representation its cells need raises an error naming the file, the side and the schema field
- `sckitflow.core.surrogate.SurrogatePotential`: a differentiable potential of continuous condition covariates through a trained inference method, averaged over the groups of its context data (`adata`), which it reads once per device onto the inference module's device. It takes a `readout` and a `potential`; `MeanSamplesReadoutConfig` and `SquaredErrorPotentialConfig` build the common ones
- Basic tool, preprocessing and plotting functions
- `docs/scripts/fit_cfm.py` trains a run from one YAML spec; install with the `scripts` extra
- `RunConfig.build` and `load_run` take `control_adata`, and `RunConfig.build` takes `metrics`

### Changed

- Flow method configs take only configs: live probability paths and raw sampler callables are rejected
- Time and noise samplers are separate families (`TimeSamplerConfig`, `NoiseSamplerConfig`)

### Fixed

- Gromov-Wasserstein and fused Gromov-Wasserstein couplings default to `scale_cost="mean"`, like the linear ones: unscaled float32 costs made entropic GW underflow to an all-zero plan. A batch whose plan is all zero still falls back to random pairs, but now warns (`RuntimeWarning`) instead of only logging
- Pure Gromov-Wasserstein matching (`use_linear=False`) no longer fails on the missing linear term ([#145](https://github.com/theislab/sckitflow/issues/145))
