# CFM Panel Sweep — Runbook

Random search over 5 adatas × 3 epsilon × 3 alpha × 2 scale_cost = 90-cell
space. Optuna `RandomSampler` + SLURM (`submitit_slurm`). Metric: `test/edist`.
`n_trials` (default 30) controls how many cells are drawn from the 90.

---

## 0. Setup

```bash
conda activate scflow13
pip install hydra-core hydra-submitit-launcher hydra-optuna-sweeper optuna
cd /ictstr01/home/icb/lorenzo.consoli/repos/sckitflow
```

---

## 1. Prepare the five pre-split adatas

Each carries `obs["split"]` = `train`/`test` (80/20 sklearn split).
`fit_cfm.py` reuses the existing column and skips its own splitter.

```bash
for seed in 0 1 2 3 4; do
  python docs/scripts/prepare_panel_adata.py \
    paths.old_adata=/abs/path/to/old_panel.h5ad \
    paths.new_adata=/abs/path/to/new_panel.h5ad \
    save.output_dir=/lustre/.../data/out/sweeps \
    prepare.split_seed=$seed
done
```

Verify:

```bash
ls -1 /lustre/.../data/out/sweeps/adata_concat_seed*.h5ad   # expect 5 files
```

---

## 2. Smoke test — 3 levels

The smoke config inherits the real sweep and sets `n_trials=1`, `n_jobs=1`,
`max_epochs=1`, tiny loader/inference sizes, and writes to `/tmp`. No CLI
grid values needed — the grid lives in the parent YAML's `choice(...)` block.

### Level 1: config resolves (5 s, no execution)

```bash
python docs/scripts/fit_cfm.py \
  --config-name panel_imputation_hparams_smoke \
  --cfg hydra --resolve 2>&1 | grep -B1 -A3 "RandomSampler"
```

Expected:

```yaml
      _target_: optuna.samplers.RandomSampler
      _partial_: false       # ← CRITICAL — must be false, not true, not absent
```

The other two `_target_`s to verify (either in the same command or with
`grep "_target_"`):

```yaml
_target_: hydra_plugins.hydra_submitit_launcher.submitit_launcher.SlurmLauncher
_target_: hydra_plugins.hydra_optuna_sweeper.optuna_sweeper.OptunaSweeper
```

**If `_partial_` is `true` or absent, do not proceed** — the smoke will
crash with:

```
ConfigAttributeError: Key 'infer_relative_search_space' not in 'SamplerConfig'
```

This is a `hydra-optuna-sweeper` merge bug: the plugin's default
`TPESamplerConfig` leaves `_partial_: true` in place and Hydra passes a raw
`SamplerConfig` dataclass to Optuna instead of an instantiated sampler.

Fix, in order (try the cheapest first):

1. **CLI force** (no file edit, usually works):
   add `'+hydra/sweeper/sampler=random'` to the launch command

2. **Upgrade the plugin**:
   ```bash
   pip install --upgrade --force-reinstall hydra-optuna-sweeper optuna hydra-core
   ```
   then re-run the Level 1 check

3. **Inline `_partial_: false`** (bulletproof): remove
   `override /hydra/sweeper/sampler: random` from the YAML's `defaults:`
   list, and add directly under `hydra.sweeper:`:

   ```yaml
   hydra:
     sweeper:
       _target_: hydra_plugins.hydra_optuna_sweeper.optuna_sweeper.OptunaSweeper
       direction: minimize
       n_jobs: 30
       n_trials: 30
       sampler:
         _target_: optuna.samplers.RandomSampler
         _partial_: false
       params:
         # (existing choice() block unchanged)
   ```

### Level 2: 1 cell via SLURM (~3 min)

```bash
python docs/scripts/fit_cfm.py \
  --config-name panel_imputation_hparams_smoke \
  --multirun \
  2>&1 | tee /tmp/cfm_smoke.log
```

Console prints `[HYDRA] Launching 1 jobs locally` then
`[HYDRA] #0 : paths.target_adata=...,coupling.epsilon=...,...` — that's the
cell `RandomSampler` drew. It varies between runs by design.

Watch: `squeue -u $USER` → one `cfm-panel-seed-0` job.

### Level 3: three pass checks (~10 s after drain)

```bash
# (a) clean exit
sacct -u $USER --starttime today --format=JobID,State,ExitCode | grep cfm
#    expect: COMPLETED 0:0

# (b) sampled cell landed in the run's config
jq '.coupling, .paths.target_adata' /tmp/cfm_sweep_smoke/seed_0/save/config.json
#    expect: epsilon ∈ {0.5,1.0,2.0}, alpha ∈ {0.25,0.5,0.75},
#            scale_cost ∈ {mean,median}, adata_concat_seed{0..4}.h5ad

# (c) test/edist reached W&B
jq '."test/edist"' /tmp/cfm_sweep_smoke/seed_0/logs/wandb/wandb/offline-run-*/files/wandb-summary.json
#    expect: finite float (not null, not Infinity)
```

(b) can't be precomputed under random sampling. If in doubt, run the smoke
**twice** and diff:

```bash
cp /tmp/cfm_sweep_smoke/seed_0/save/config.json /tmp/cfg1.json
rm -rf /tmp/cfm_sweep_smoke
python docs/scripts/fit_cfm.py \
  --config-name panel_imputation_hparams_smoke --multirun >/dev/null 2>&1
jq '.coupling' /tmp/cfg1.json /tmp/cfm_sweep_smoke/seed_0/save/config.json
# different across runs = sampler working + injection working
# identical twice     = sweeper NOT injecting (grid being ignored)
```

All three green → cleanup and proceed:

```bash
rm -rf /tmp/cfm_sweep_smoke /tmp/cfg1.json
```

---

## 3. Real sweep — 30 random draws (default)

Launched as a **single `sbatch` job**: the Hydra driver runs inside SLURM (not on the login node), so it survives disconnect without `tmux`, and it spawns the 30 Optuna trials as child submitit jobs on the GPU partition configured in the YAML. No script file is written — `--wrap` carries the command inline.

```bash
sbatch --job-name=cfm-sweep \
       --partition=cpu_p \
       --qos=cpu_normal \
       --cpus-per-task=2 \
       --mem=8G \
       --time=48:00:00 \
       --chdir=/ictstr01/home/icb/lorenzo.consoli/repos/sckitflow \
       --output=/tmp/cfm_sweep_%j.log \
       --wrap="bash -lc 'source activate scflow13 && \
                python docs/scripts/fit_cfm.py \
                  --config-name panel_imputation_hparams \
                  --multirun'"
```

The parent config sets `n_trials: 30, n_jobs: 30`. Each trial is an independent draw from the 90-cell space; repeats and gaps are expected (that's random search, not grid search).

### Why `sbatch` instead of `tmux`

| Concern | How the `sbatch` wrapper handles it |
|---|---|
| Driver dying on disconnect | Runs as a SLURM job — survives SSH drops, laptop sleep, VPN flaps |
| `conda activate` on a compute node | `bash -lc 'source activate scflow13 && …'` restores the shell profile and activates the env inside the job |
| Relative paths (`config/`, `save.run_dir`, loggers) | `--chdir=` pins the launch directory; combined with `hydra.job.chdir: false` in the YAML, all relative paths resolve from the repo root |
| Logs | `--output=/tmp/cfm_sweep_%j.log` — `%j` is the driver's SLURM job id |

**Driver resources are tiny:** 1–2 CPUs and 8 G are plenty. The driver only launches and monitors submitit jobs. The **trials** request GPUs via `hydra.launcher.{partition,qos,gres,constraint}` in the YAML (`gpu_p` / `gpu_normal` / `gpu:1` / `h100_80gb|a100_80gb`), unchanged by this wrapper.

### Confirm the driver + 30 trials are queued

```bash
squeue -u $USER | grep cfm-sweep      | wc -l   # expect: 1   (driver)
squeue -u $USER | grep cfm-panel-seed | wc -l   # expect: 30  (trials, once driver runs)
```

Right after submission only the driver appears; the 30 `cfm-panel-seed-*` jobs are submitted by the driver a few seconds into its run. Follow it with:

```bash
tail -f /tmp/cfm_sweep_<driver_jobid>.log
```

The driver exits (`COMPLETED`) only after all 30 trials finish, so its state is a reliable end-of-sweep signal:

```bash
sacct -j <driver_jobid> --format=JobID,State,ExitCode,Elapsed
```

### Widen coverage without editing the file

```bash
sbatch --job-name=cfm-sweep-90 \
       --partition=cpu_p --qos=cpu_normal \
       --cpus-per-task=2 --mem=8G --time=48:00:00 \
       --chdir=/ictstr01/home/icb/lorenzo.consoli/repos/sckitflow \
       --output=/tmp/cfm_sweep90_%j.log \
       --wrap="bash -lc 'source activate scflow13 && \
                python docs/scripts/fit_cfm.py \
                  --config-name panel_imputation_hparams \
                  --multirun \
                  hydra.sweeper.n_trials=90 \
                  hydra.sweeper.n_jobs=90'"
```

90 draws from a 90-cell space ≈ 63 % distinct cells (birthday problem). For true exhaustive coverage, switch to the `basic` sweeper (grid) or `GridSampler` instead.

### Monitoring reminder (§4 applies unchanged)

`grep cfm` now matches **both** the driver (`cfm-sweep…`) and the trials (`cfm-panel-seed-*`). To watch only trial throughput, filter by `cfm-panel-seed`:

```bash
squeue -u $USER -o "%.8i %.30j %.8T" | grep cfm-panel-seed | awk '{print $3}' | sort | uniq -c
```

If you see `1 RUNNING, 29 PENDING`, that's the QoS `MaxJobsPU` throttle described in §4 — the driver is fine, the trials are being rate-limited by SLURM, not by Hydra.

---

## 4. Monitor

```bash
# SLURM status
squeue -u $USER -o "%.8i %.30j %.8T %.10M %R" | grep cfm

# Job completion
sacct -u $USER --starttime today --format=JobID,State,ExitCode,Elapsed | grep cfm

# Live metric of one trial
tail -f /lustre/.../sweeps/seed_7/logs/csv/*/metrics.csv
```

### If jobs run one at a time instead of in parallel

First, check that the smoke wasn't the test — a 1-cell smoke has nothing to
parallelize. Verify with an actual multi-cell launch (`n_trials=4`).

Then check what SLURM is actually doing:

```bash
squeue -u $USER -o "%.8i %.30j %.8T" | grep cfm | awk '{print $3}' | sort | uniq -c
```

- `4 PENDING, 1 RUNNING` → QoS `MaxJobsPU` throttle (most common). Check:

  ```bash
  sacctmgr show qos gpu_preemptible format=Name,MaxJobsPU,MaxSubmitJobsPU,GrpTRES
  ```

  If `MaxJobsPU=1`, switch QoS in the YAML (`qos: gpu` for non-preemptible)
  or ask admin to raise the cap. `array_parallelism` sets the array throttle
  Hydra requests; QoS caps what SLURM grants.

- `4 PENDING` all with reason `ReqNodeNotAvail` → the `constraint:` line is
  unsatisfiable. Comment it out in the YAML and relaunch.

---

## 5. Collect results

Rank all completed cells by `test/edist` (works regardless of how many
completed — 30 default, or more if you widened `n_trials`):

```bash
SWEEP_ROOT=/lustre/.../data/out/sweeps

for d in "$SWEEP_ROOT"/seed_*/; do
  f=$(find "$d" -name wandb-summary.json 2>/dev/null | head -1)
  edist=$(jq -r '."test/edist"' "$f" 2>/dev/null)
  eps=$(jq -r '.coupling.epsilon' "$d/save/config.json" 2>/dev/null)
  alp=$(jq -r '.coupling.alpha' "$d/save/config.json" 2>/dev/null)
  sc=$(jq -r '.coupling.scale_cost' "$d/save/config.json" 2>/dev/null)
  ad=$(jq -r '.paths.target_adata // ""' "$d/save/config.json" 2>/dev/null | xargs -n1 basename 2>/dev/null)
  [ -n "$edist" ] && [ "$edist" != "null" ] && \
    printf "%-10s edist=%-10s eps=%-4s alpha=%-4s sc=%-6s adata=%s\n" \
      "$(basename "$d")" "$edist" "$eps" "$alp" "$sc" "$ad"
done | sort -t= -k2 -g
```

The lowest `test/edist` is the winner. Its `seed_<N>/save/` is a full `Model`
loadable via `Model.from_pretrained`.

Sync W&B offline runs:

```bash
wandb sync --sync-all "$SWEEP_ROOT"
```

All runs appear grouped as `panel-adata-coupling-sweep` (from
`loggers.wandb.init_kwargs.group`), tagged by seed, plotted against
`test/edist`.

---

## 6. Extending the search space

The parent YAML's `hydra.sweeper.params` block has commented-out axes for
neural velocity and condition encoder widths. Uncomment **one** at a time,
then bump `n_trials` proportionally:

```yaml
# inside hydra.sweeper.params:
module.state_encoder_output_dim: choice(16, 32)          # ×2
module.condition_encoder_output_dim: choice(32, 64)      # ×2
```

| Axis added | Space size | Suggested `n_trials` |
|---|---|---|
| (baseline) | 90 | 30 |
| `state_encoder_output_dim` | 180 | 60 |
| `condition_encoder_output_dim` | 180 | 60 |
| Both | 360 | 120 |

Random search efficiency drops as the space grows — for very wide spaces,
switch to `TPESampler` (drop the `override /hydra/sweeper/sampler: random`
line and use the plugin's default) so trials cluster around good regions.

---

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `ConfigAttributeError: Key 'infer_relative_search_space' not in 'SamplerConfig'` | `hydra-optuna-sweeper` merge bug — see §2 Level 1 fixes (CLI force / upgrade / inline `_partial_: false`) |
| `Key 'partition' not in 'BasicLauncherConf'` | You passed `hydra/launcher=basic` — drop it; the YAML sets `submitit_slurm` |
| `Key 'n_jobs' not in 'BasicSweeperConf'` | You mixed a `basic` sweeper override with Optuna-only keys — remove `override /hydra/sweeper: basic` (or remove `n_jobs`/`n_trials`/`params`) |
| `Could not find 'hydra/sweeper/optuna'` | Install `hydra-optuna-sweeper`: `pip install hydra-optuna-sweeper optuna` |
| `IndexError` at `val_names[0]` | `obs["split"]` has no `test` labels — re-run §1 and verify |
| `File exists: .../save` | A preempted trial left a partial dir — `rm -rf` it, or set `run.allow_overwrite: True` |
| All jobs `PENDING` (`ReqNodeNotAvail`) | Comment out `constraint:` in the yaml |
| Jobs queue one-at-a-time | QoS `MaxJobsPU` cap — see §4 |
| `Sinkhorn did not converge` | Harmless unless edist plateaus — try larger `coupling.epsilon` |
| Smoke passes but real sweep has `File exists` on retry | `qos: gpu_preemptible` preempted the trial mid-write — `rm -rf seed_<N>/save` and relaunch, or set `allow_overwrite: True` |

---

## 8. File tree after a completed sweep

```
/lustre/.../data/out/sweeps/
├── adata_concat_seed{0..4}.h5ad       # inputs from prepare_panel_adata.py
├── seed_0/                             # one dir per Optuna trial
│   ├── save/                           # Model.from_pretrained target
│   │   ├── config.json                 # sampled hparams for this trial
│   │   ├── module/model.safetensors    # model weights
│   │   ├── transform.json
│   │   └── transform_params.npz
│   ├── checkpoints/                    # Lightning ModelCheckpoint outputs
│   └── logs/
│       ├── wandb/wandb/offline-run-*/  # to sync after sweep
│       └── csv/                        # metrics.csv with per-epoch test/edist
├── seed_1/ … seed_29/                  # (or seed_0..seed_N for n_trials=N)
```
