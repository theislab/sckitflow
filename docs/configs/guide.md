# CFM configurations — README

Reference for the five YAML configs shipped with the CFM panel-imputation
pipeline. Each config is a **Hydra** config inheriting from a parent via the
`defaults:` list, so every file only lists what it *adds or overrides* — the
rest comes from the parent.

---

## Inheritance tree

```
fit_cfm.yaml                                ← base config (all defaults)
└── fit_cfm_panel.yaml                      ← panel setup (fused-GW, resnet1d)
    └── panel_imputation_hparams.yaml       ← sweep on top of panel setup
        └── panel_imputation_hparams_smoke.yaml

fit_cfm_panel_smoke.yaml                    ← fast smoke test (see §5 caveat)
```

> **Note on `panel_imputation`:** `panel_imputation_hparams.yaml` and
> `fit_cfm_panel_smoke.yaml` both inherit from a config named
> `panel_imputation` which is **not in this bundle**. In practice that should
> be `fit_cfm_panel` — see §6 for the fix.

---

## 1. `fit_cfm.yaml` — base config

**Purpose:** complete defaults for the CFM training pipeline. Every top-level
section is read by `docs/scripts/fit_cfm.py` as `config.<section>.<key>`.

**Inherits from:** none.

**When to use:** only as a parent (via `defaults:`). Direct invocation works
but requires overriding `paths.target_adata` at minimum.

**Key defaults:**

| Section | Default | Meaning |
|---|---|---|
| `paths.target_adata` | `/abs/path/to/target.h5ad` | must be overridden |
| `probability_path.name` | `linear_gaussian` | alternatives: `linear_dirac`, `schrodinger_bridge`, `variance_preserving` |
| `splitter` | `batch`-grouped 80/20 split | writes `obs["split"]` |
| `datamanager.sample_rep` | `X_pca` | input representation for the VF |
| `datamanager.control_values_dict` | `{}` | unpaired by default |
| `transform.name` | `identity` | alternatives: `pca`, `zscore`, `whitening` |
| `module` | 128-dim state/time, `concat` conditioning, 512×512 MLPs | **no** source encoder (linear-coupling only) |
| `coupling.method` | `independent` | alternatives: `exact`, `sinkhorn`, `partial`, `unbalanced`, `gromov-wasserstein`, `fused` |
| `optimizer` | `Adam(lr=1e-4, wd=1e-5)` | notebook uses `AdamW`; override for exact parity |
| `inference.n_steps` / `n_samples` | `50` / `1000` | euler solver |
| `trainer.max_epochs` | `100` | `accelerator: auto` |
| `run.allow_overwrite` | `false` | prevents accidental clobbering |

**Source-encoder rule:** `module.source_encoder_*` is `null` by default.
Only **quadratic** couplings (`gromov-wasserstein`, `fused`) in a **paired**
schema need it. Enable per-run with:

```bash
module.source_encoder_output_dim=32 \
module.source_encoder_mlp_kwargs.hidden_dims=[32] \
module.source_encoder_mlp_kwargs.activation_cls=silu
```

**Usage:**

```bash
python docs/scripts/fit_cfm.py \
  paths.target_adata=/abs/path/to/target.h5ad \
  save.run_dir=/abs/path/to/out
```

---

## 2. `fit_cfm_panel.yaml` — panel imputation setup

**Purpose:** reproduces the `genot_cross_panel` notebook setup — Fused
Gromov-Wasserstein coupling, old panel = source/control, new panel = target,
small MLP with `resnet1d` conditioning and an explicit source encoder.

**Inherits from:** `fit_cfm.yaml` via `defaults: [fit_cfm, _self]`.

**When to use:** any single-run panel imputation (no sweeping). This is the
"real" run config.

**What it overrides vs. the base:**

| Section | Change | Why |
|---|---|---|
| `paths.target_adata` | absolute path to `adata_concat.h5ad` | panel-concat dataset |
| `splitter` | `group_keys: ["dataset"]`, `control_key/value: dataset/old panel`, `always_train_keys: []` | split by panel, old panel is source |
| `datamanager` | `sample_rep: X_zscore`, `panel` condition, `rep_lin: X_lin`, `source_rep_quad: X_quad_src`, `target_rep_quad: X_quad_tgt`, `use_linear: true` | fused-GW needs linear + quadratic reps from the notebook |
| `module` | tiny: 16-dim state/time/source, `resnet1d` conditioning, 64×64 decoder, 256 time features, `ott-jax` time encoding, **source encoder ON** | matches the printed model repr in the notebook |
| `probability_path` | `linear_dirac` | deterministic path used in the notebook |
| `coupling` | `fused`, `scale_cost: median`, `epsilon: 1.0`, `max_iter: 1000`, `tol: 1e-9`, `alpha: 0.5` | fused-GW defaults from the notebook |
| `optimizer` | `AdamW(lr=1e-4)` | notebook uses AdamW |
| `metrics.mmd.gammas` | `null` | let the kernel pick its own bandwidths |
| `trainer` | `accelerator: cuda`, absolute `default_root_dir` | GPU + non-relative paths |
| `run.method_seed` | `0` | reproducible |
| `run.allow_overwrite` | `true` | cheap iteration |
| All writer paths | absolute, under `.cache/panel_int/` | work from any CWD |

**Usage:**

```bash
python docs/scripts/fit_cfm.py --config-name fit_cfm_panel \
  paths.target_adata=/abs/path/to/adata_concat.h5ad \
  save.run_dir=/abs/path/to/out
```

---

## 3. `fit_cfm_panel_smoke.yaml` — fast smoke test for the panel config

**Purpose:** exercises the **full model construction + one forward/backward**
in seconds instead of hours, so you catch config/wiring bugs before committing
to a real run.

**Inherits from:** `panel_imputation` (see caveat in §6 — should be
`fit_cfm_panel`).

**When to use:** after every edit to `fit_cfm_panel.yaml` or to
`docs/scripts/fit_cfm.py`, before launching a real run.

**What it overrides:**

| Section | Change | Why |
|---|---|---|
| `module` | even smaller: 4-dim state/time/source, 8-unit MLPs, 16 time features | build + forward/backward in <1 s |
| `coupling` | `epsilon: 0.1`, `max_iter: 20`, `tol: 1e-3` | Sinkhorn on a full batch is the expensive step |
| `datamodule` | `n_train_steps: 5`, `batch_size: 16` | smallest stream that still pushes a batch |
| `inference` | `n_steps: 3`, `n_samples: 8` | cheap validation ODE |
| `trainer` | `fast_dev_run: true`, `enable_progress_bar: false` | Lightning: one batch, no ckpt/loggers |
| `save.run_dir` | `./runs/cfm_panel_smoke` | local, not lustre |
| `run.allow_overwrite` | `true` | cheap iteration |
| `loggers` | local `save_dir` | avoids slow WandbLogger init |

**`fast_dev_run: true`** disables checkpointing and loggers, but
`save_pretrained` at the end of `main()` **still runs**, so you get
`config.json` + `model.safetensors` + `transform.json` in `save.run_dir` —
a real artifact from a sub-minute run.

**Usage:**

```bash
python docs/scripts/fit_cfm.py --config-name fit_cfm_panel_smoke
```

---

## 4. `panel_imputation_hparams.yaml` — 30-trial random sweep

**Purpose:** sweeps over **5 adatas × 3 ε × 3 α × 2 `scale_cost` = 90 cells**
using Optuna `RandomSampler` + `submitit_slurm`, minimising `test/edist`.
Default `n_trials=30, n_jobs=30` draws 30 of those 90.

**Inherits from:** `panel_imputation` (see §6).

**When to use:** hyperparameter search over the fused-GW coupling knobs and
the input split seed. This is the config driven by §3 of the runbook.

**What it adds / overrides:**

| Section | Value | Why |
|---|---|---|
| `defaults` | `+submitit_slurm`, `+optuna`, `+random` sampler | SLURM scheduling + Optuna search |
| `hydra.job.chdir` | `false` | relative paths resolve from repo root |
| `hydra.sweep.dir` | absolute path under `data/out/sweeps` | persistent sweep tree |
| `hydra.sweep.subdir` | `seed_${hydra.job.num}` | one dir per trial |
| `hydra.launcher` | `gpu_p` / `gpu_normal` / `gpu:1` / `h100_80gb\|a100_80gb` / 64 GB / 6 CPU / 72 h / `array_parallelism: 10` | trial resource request |
| `hydra.sweeper._recursive_` | `true` | **critical fix** — without it Optuna receives a raw `SamplerConfig` and crashes with `Key 'infer_relative_search_space' not in 'SamplerConfig'` |
| `hydra.sweeper.direction` | `minimize` | lower edist is better |
| `hydra.sweeper.n_trials` | `30` | random draws from the 90-cell space |
| `hydra.sweeper.n_jobs` | `30` | parallel SLURM jobs |
| `hydra.sweeper.params` | `paths.target_adata ∈ {seed0..4}`, `coupling.epsilon ∈ {0.5,1,2}`, `coupling.alpha ∈ {0.25,0.5,0.75}`, `coupling.scale_cost ∈ {mean,median}` | the 90-cell grid |
| `save.run_dir` | `${hydra:runtime.output_dir}/save` | per-trial |
| `callbacks.early_stopping` | monitor `test/edist`, `patience: 10` | early stop on the swept metric |
| `callbacks.model_checkpoint` | monitor `test/edist` | save best-edist ckpts |
| `trainer` | `max_epochs: 30`, `default_root_dir` per-trial | bounded sweep runtime |
| `loggers.wandb` | offline, name `cfm-panel-seed-${hydra.job.num}`, group `panel-adata-coupling-sweep` | grouped W&B |
| `run.allow_overwrite` | `false` | never clobber a completed trial |

> ⚠️ **Do not add a manual `sampler:` block** under `hydra.sweeper` — the
> `override /hydra/sweeper/sampler: random` in `defaults:` already selects
> the sampler, and adding one reintroduces the `SamplerConfig` crash.

**Usage (single `sbatch` job — see runbook §3):**

```bash
sbatch --job-name=cfm-sweep \
       --partition=cpu_p --qos=cpu_normal \
       --cpus-per-task=2 --mem=8G --time=48:00:00 \
       --chdir=/ictstr01/home/icb/lorenzo.consoli/repos/sckitflow \
       --output=/tmp/cfm_sweep_%j.log \
       --wrap="bash -lc 'source activate scflow13 && \
                python docs/scripts/fit_cfm.py \
                  --config-name panel_imputation_hparams \
                  --multirun'"
```

---

## 5. `panel_imputation_hparams_smoke.yaml` — 1-trial sweep smoke

**Purpose:** validates the full **Optuna + submitit_slurm** wiring end-to-end
in ~3 min, without touching the real sweep tree.

**Inherits from:** `panel_imputation_hparams.yaml` via
`defaults: [panel_imputation_hparams, _self]`.

**When to use:** before every real sweep launch — this is the config that
catches the `SamplerConfig` bug, a broken `constraint:`, or a missing
`obs["split"]`.

**What it overrides:**

| Section | Value | Why |
|---|---|---|
| `hydra.sweep.dir` | `/tmp/cfm_sweep_smoke` | throwaway tree |
| `hydra.sweeper.n_trials` | `1` | single draw from the 90-cell space |
| `hydra.sweeper.n_jobs` | `1` | one SLURM job |
| `datamodule.n_train_steps` | `20` (from 100) | fast epoch |
| `inference.n_steps` / `n_samples` | `3` / `8` | cheap validation ODE |
| `trainer.max_epochs` | `1` | stop after one epoch |
| `run.allow_overwrite` | `true` | iterate cheaply |

The grid (`paths.target_adata`, `coupling.*`) is **inherited** from
`panel_imputation_hparams.yaml` — do not pass grid values on the CLI; the
sweeper samples one cell.

**Usage:**

```bash
sbatch --job-name=cfm-smoke \
       --partition=cpu_p --qos=cpu_normal \
       --cpus-per-task=1 --mem=4G --time=00:30:00 \
       --chdir=/ictstr01/home/icb/lorenzo.consoli/repos/sckitflow \
       --output=/tmp/cfm_smoke_%j.log \
       --wrap="bash -lc 'source activate scflow13 && \
                python docs/scripts/fit_cfm.py \
                  --config-name panel_imputation_hparams_smoke \
                  --multirun'"
```

---

## 6. Known inconsistencies in this bundle

Two things to fix before treating the bundle as self-consistent:

1. **`fit_cfm_panel_smoke.yaml` inherits the wrong parent.** Its `defaults:`
   says `panel_imputation`, but its header says *“Inherits `fit_cfm_panel`
   verbatim”*. `panel_imputation` is not in this bundle. Change to:

   ```yaml
   defaults:
     - fit_cfm_panel
     - _self_
   ```

2. **`panel_imputation_hparams.yaml` inherits `panel_imputation`**, which is
   also not in this bundle. If `panel_imputation.yaml` is just
   `fit_cfm_panel.yaml` under another name, point both configs at
   `fit_cfm_panel` and delete the phantom name; otherwise ship the missing
   file.

3. **Module-path drift in docstrings.** `fit_cfm.yaml` refers to
   `python -m <pkg>.fit_cfm`; the other configs refer to
   `python docs/scripts/fit_cfm.py`. Standardise on one — the sweep launcher
   uses `docs/scripts/fit_cfm.py`.

---

## 7. Quick-reference table

| Config | Inherits from | Trials | Coupling | SLURM | Writes to | Runtime (approx) |
|---|---|---|---|---|---|---|
| `fit_cfm` | — | 1 | `independent` | no | `./runs/cfm_run` | hours |
| `fit_cfm_panel` | `fit_cfm` | 1 | `fused` | no | `.cache/panel_int/` | hours |
| `fit_cfm_panel_smoke` | `panel_imputation`* | 1 | `fused` (tiny) | no | `./runs/cfm_panel_smoke` | <1 min |
| `panel_imputation_hparams` | `panel_imputation`* | 30 | `fused` | yes | `data/out/sweeps/seed_*` | 30 × hours, parallel |
| `panel_imputation_hparams_smoke` | `panel_imputation_hparams` | 1 | `fused` (tiny) | yes | `/tmp/cfm_sweep_smoke` | ~3 min |

\* parent missing from this bundle — see §6.

---

## 8. Rules of thumb

- **Editing the pipeline?** Run `fit_cfm_panel_smoke` after every change — it
  exercises the same construction path as a real run, in under a minute.
- **Editing the sweep?** Run `panel_imputation_hparams_smoke` after every
  change — it catches Optuna/sweeper/launcher bugs that a single-run smoke
  cannot.
- **Never edit `fit_cfm.yaml`** to prototype a panel idea — put the change in
  a child config so the base defaults stay a stable reference.
- **Never pass grid values on the CLI** when launching a sweep — the sweeper
  samples them; passing them alongside a `choice(...)` grid silently
  overrides the search.
- **`run.allow_overwrite`** is `false` in the base and real-sweep configs on
  purpose. Set to `true` only on throwaway runs, otherwise a preempted trial
  leaves a partial `save/` that a relaunch will refuse to overwrite (see
  runbook §7).
