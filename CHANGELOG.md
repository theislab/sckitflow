# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [Unreleased]

### Changed

- `FlowDataModule` holds its `adata` for its whole life: `attach`, `set_predict_data`, `predict_dataloader` and its `control_adata` are removed. `load_from_checkpoint(path, adata)` builds it with the data, and `predict_adata` builds its eval loader per call.
- Every `AnnData` argument of the data stack (`DataManager`, `FlowDataModule`, the loaders, `predict_adata`) also takes a list of them, read as one dataset; they must agree on dims and feature names, and the `.uns` tables the schema reads are merged across them (an entry in several must match). Requires scfit 0.0.11
- `control_adata` is removed everywhere (`DataManager.get_dataloaders` / `get_eval_loader`, the loaders, `predict_adata`). Controls are drawn from the streamed data by the schema, and the loader holds just them in memory.

### Added

- Basic tool, preprocessing and plotting functions
