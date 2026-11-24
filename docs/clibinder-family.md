# sckitflow as a clibinder family: migration plan

Status: agreed 2026-11-24, not started. Goal: sckitflow trains, saves, predicts and evaluates through
[clibinder](https://github.com/theislab/clibinder)'s four functions, so the flowrun runner
(`flowrun-torch-extras`, package `flowrun`) can be archived.

clibinder calls a family through:

```python
fit(spec: dict, data: dict, run: dict) -> dict[str, float]   # writes run["out_dir"]
load(path)                                                   # checks the saved folder
predict(path, data, out) -> None
evaluate(preds, data) -> dict[str, float]
```

`data = {"paths": [...], "split": <split file>, "subset": ...}`; `run = {"out_dir", "seed", "devices", "precision",
"max_time", "resume_from", "opts"}`. The family opens the stores, joins its rows to the split file, owns its
training loop, and never downloads anything (pretrained weights come in as a local folder).

## Decisions

1. **Spec shape: `{model, train}`.** `model` is a `ModelConfig` spec, unchanged; it alone is written to
   `config.json`. `train` is a new `TrainConfig`: optimizer, batch size, steps, loader kwargs, metrics, monitor,
   early stopping, Trainer pass-through kwargs, W&B. It goes into the run's `run.yaml`, not into the model.
2. **No test metrics in fit.** `BestMetrics` keeps `val/...` only; test is scored by `predict` + `evaluate`. Old
   flowrun runs logged `test/` at the best validation epoch, so those numbers don't line up one-to-one.
3. **Preprocessing is part of the model.** `ModelConfig.preprocessing: PreprocessingConfig | None`; PCA, z-score
   and whitening are fitted in `fit` on train rows and saved next to the weights. Nothing is written to the input
   stores.
4. **Shared/own features are selected inside the data manager** (by column name), not precomputed into `obsm`.
5. **Base:** stacked on `feat/datamanager-fields` (theislab/sckitflow#231), because step 5 extends its
   `anndata.acc` references. #231 itself waits on theislab/scfit#31. Nothing here needs #230 (loss guidance).
6. This file is where the plan lives.

## Where flowrun's code goes

Paths in `flowrun-torch-extras/src/flowrun/`.

| flowrun | Goes to | Note |
|---|---|---|
| `fit_cfm.py` builder chains (377–493), `resolve_omegaconf…`, `drop_nulls` | deleted | the `model` spec already builds these through `ModelConfig.build` |
| `fit_cfm.py` Trainer, loggers, callbacks, save, returned metric (527–708) | `sckitflow/trainer/_build.py` + the adapter | see step 3 and 6 |
| `fit_cfm.py` split block (290–318), `split.split_labels` | deleted | clibinder writes split files; the adapter joins them |
| `fit_cfm.read_zarr_input` | deleted | imports `scfit.data._io.load_backed_adata`, which current scfit no longer has |
| `fit_cfm.enable_zarrs` | adapter, read from `run["opts"]` | a speed knob |
| `fit_cfm.key_parameters`, sweep configs, `docs/sweeps.md` | deleted | sweeps are Nextflow channels |
| `_callbacks.BestMetrics` | `sckitflow/trainer/_callbacks.py` | `val/` only (decision 2) |
| `preprocess.preprocess/apply/inverse` | `sckitflow/preprocessing/` (restored) | fitted in `fit`, saved with the model (decision 3) |
| `preprocess.write_preprocessing`, `_store.py` | deleted | they wrote into the input files |
| `panels.from_state` | data manager (step 5) | replaced by `features: shared \| own` |
| `panels.prepare_panels`, `main` | deleted | the project's prepare step does this |
| `split.random_cells`, `split.shared_source_combination` | clibinder `splitters` | generic; port with identical counting so held-out sets don't change |
| `split.shared_source_holdout` | deleted | clibinder's `groups` covers it |
| `evaluate.py` | CITESeq-Imputation (`scmpute/tl`) | panel-specific |
| `experiment/cellflow*.yaml`, `genot.yaml` | `docs/configs/` | `cellflow.yaml` is already there |
| `experiment/flow_cyto_panel_alignment*.yaml` | CITESeq-Imputation | project configs |
| `tests/test_fit_cfm.py` (745 lines) | `tests/test_family.py` | port the axis, coupling and schema-pair parametrisations |

## Steps, in order

### 1. Keep past splits reproducible (clibinder)

Add `clibinder.splitters.from_column(units, *, column)`, keyed per row, which reads an existing obs column. Run it
once per old seed over the current `FlowCytometry/out/data` stores (`obs["split_<seed>"]`) to export the old
held-out cells as split files. flowrun's labels (`train/val/test/all`) are already clibinder's. Do this before any
flowrun code is deleted.

### 2. Namespaced component ids

`ModelConfig` is registered as `"model"`, so clibinder (which imports the first part of `type`) would `import
model`. Rename the ids to `sckitflow.<name>` (`sckitflow.model`, `sckitflow.data_manager`, …), with the old ids
accepted when loading existing `config.json` files.

### 3. Trainer from `run` and `BestMetrics`

- `sckitflow/trainer/_callbacks.py`: `BestMetrics(monitor, mode)`, `val/` metrics at the best monitored epoch.
- `sckitflow/trainer/_build.py`: `build_trainer(train: TrainConfig, run: dict) -> lightning.Trainer`.
  - From `run`: `devices`, `precision`, a `Timer` for `max_time`, `ModelCheckpoint(dirpath=out_dir/checkpoints,
    save_last=True)`, and `ckpt_path` = `resume_from/last.ckpt` when present.
  - From `train`: early stopping, gradient accumulation, LR monitor, the remaining Trainer kwargs (pass-through),
    and W&B if `train.wandb` is set (finished in `finally`).
  - Raise if the monitored metric was never logged (flowrun's lines 684–689).

No behaviour change yet; testable alone.

### 4. Preprocessing in the model

- Restore `sckitflow/preprocessing/` from flowrun's `preprocess.py` without `_store` and `write_preprocessing`.
- `ModelConfig.preprocessing: PreprocessingConfig | None` (`pca`, `n_hvg`, `hvg_flavor`, `zscore`, `whitening`,
  `epsilon`, `input`).
- `fit`: fit on the train rows (plus `all` rows), apply to every row in memory, save the fitted record as a second
  state: `save_pretrained(path, spec, {"module": ..., "preprocessing": ...})`.
- `Model.from_pretrained`: reload it and reapply it; `predict`: invert it so predictions come back in the input
  feature space.

### 5. Feature selection in the reps (data manager)

- `data/_manager.py`, `DataManagerConfig`: `rep_lin`, `source_rep_quad` and `target_rep_quad` accept
  `{rep: layers.<k>, features: shared | own | [names]}`.
  - `shared` = the markers every file has, in one fixed order; `own` = each side's markers minus the shared ones.
  - Resolved when the model is built; the explicit name lists replace `shared`/`own` before `save_pretrained`, so
    a saved model always reads its training columns, even on new data.
  - All files on one side must have the same markers, or it raises.
- `data/schemas/_coupling_data_schema.py`: scfit streams the whole layer per side; the schema cuts `lin` and `quad`
  out of the batch by precomputed column indices.
- Dimensions come from the name lists' lengths, not from array shapes.

This removes `panels.from_state`, the `obsm` copies, `lin_from_state`/`quad_from_state`, and the need to hold the
stores in memory. Independent of scfit's `var=` remap (that is for scConcept and the census data).

### 6. The adapter

New `sckitflow/_family.py`, re-exported as `sckitflow.fit`, `load`, `predict`, `evaluate`:

```python
def fit(spec, data, run):
    seed_everything(run["seed"])
    stores = open_stores(data["paths"])                       # backed zarr
    split = join_split(stores, data["split"])                 # clibinder.split_labels once it exists; drops "none" rows
    train_cfg = TrainConfig.from_spec(spec["train"])
    model = ModelConfig.from_spec(spec["model"]).build(stores)  # resolves features, fits preprocessing on train rows
    dm = FlowDataModule(model.manager, model.dims, stores, split_by=split, ...)
    plan = TrainingPlan(model.training_method, train_cfg.optimizer(model.module), metrics=..., seed=run["seed"])
    trainer = build_trainer(train_cfg, run)
    trainer.fit(plan, datamodule=dm, ckpt_path=last_ckpt(run))
    model.save_pretrained(run["out_dir"])
    return best_metrics(trainer)                               # {"val/edist": ..., ...}

def load(path):               return ModelConfig spec + weights from path
def predict(path, data, out): Model.from_pretrained + predict_adata on data["subset"], written as a zarr
def evaluate(preds, data):    edist / MMD per group against the subset's real cells
```

Tests in `tests/test_family.py`: the ported `test_fit_cfm.py` parametrisations through `sckitflow.fit`, plus
`clibinder.testing.check_family` once clibinder has it.

### 7. Reproduce, then archive

Run one old CITE-seq run (the fused sweep's run 5) through `clibinder-fit` on the split exported in step 1, compare
`val/` metrics, then archive `flowrun-torch-extras`.

## Open PRs this builds on

| PR | What | Needed by |
|---|---|---|
| theislab/scfit#31 | `anndata.acc` references in scfit | sckitflow#231 |
| theislab/sckitflow#231 (draft) | `anndata.acc` references for every rep, `get_required_fields` | step 5 |
| theislab/sckitflow#230 | loss-guided velocity | nothing here |
| theislab/sckitflow#212 (draft) | docs build | nothing here |

`integrate/experiments` is `main` plus #231 and #230 merged in; it is a test branch, not a base.
