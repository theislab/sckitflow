# Plan: torch-native GENOT (on the OTFM foundation)

**Branch:** `feat/refactors` · **Status:** **G1 (GENOT-L) + G2 (quadratic/GW) shipped**; G3 (options) pending.
Builds directly on the shipped torch-native OTFM (see `docs/plans/cellflow-vendor-fm.md`). Same rule: models
in torch/Lightning, the **only** JAX is the minibatch OT coupling (linear sinkhorn + quadratic/fused GW).

## ✅ G2 shipped (GENOT-Q, quadratic / fused Gromov-Wasserstein)

Coupling now routes to GW when the schema is quadratic. What landed:

- **`_quadratic_indices`** (`training/_objective.py`) — reads `src_quad`/`tgt_quad` (+ `src_lin`/`tgt_lin`
  for fused) from the batch reps and calls the existing JAX `ot_quadratic_coupling` (ott GW); the shared
  `_TorchOTObjective._couple` picks GW vs linear sinkhorn from `self._quad` (schema has `*_quad`). The
  resample reorders the state `source`/`target` exactly like the linear path — **the generated state space
  is untouched; GW only pairs cells by intra-domain structure** (source/target quad reps may differ in dim).
- Works for **OTFM and GENOT** (coupling is orthogonal to the flow); `match_method="independent"` still
  bypasses to random pairing. Seeded (reproducible) via the same `np.random.Generator`.
- **Tests:** `test_genot_quadratic_coupling_fit_predict` (GW path taken, fit/predict/reproducible) +
  `test_model_learns_conditional_translation[otfm|genot]` — a real **learning** check: after 200 steps both
  objectives move control cells in each drug's true (opposite) direction and separate the two drugs, proving
  coupling + flow + condition encoder train together.

## ✅ Stochastic condition encoder + encode-once refactor (shared-seam, both objectives)

Added in the **shared** layer, so OTFM and GENOT both get it with no per-objective code:

- **Encode-once refactor** — the condition is encoded **once** per step. `MLPVelocity.condition_stats` returns
  `(mean, logvar)`; `velocity_from_embedding(t, x, cond_embedding, source)` takes a precomputed embedding;
  `forward = condition_stats(mean) ∘ velocity_from_embedding`. The objective's shared `_encode` computes it
  once and reuses it for the velocity **and** the reg — cheaper, and *correct* for stochastic (one noise
  draw). Deterministic behavior stayed **bit-identical** (`test_flow_matching_bit_reproducible` unchanged).
- **Stochastic CE** — `SetEncoder` gains a `condition_mode`/variance head → `(mean, logvar)`; `MLPVelocity`
  exposes `condition_mode`/`is_stochastic`; the objective reparameterizes `z = mean + exp(0.5·logvar)·ε`
  (seeded `_enc_gen`) and `_encoder_reg` switches between deterministic L2 (`0.5·mean(mean²)`, gated) and the
  **VAE KL** (`0.5·mean(mean² + eᶫᵒᵍᵛᵃʳ − logvar − 1)`, always on) — matching cellflow's stochastic
  `encoder_loss`. Inference uses the mean (encoder_noise = 0). Select via `FlowMatching(condition_mode=…)`.
- **Tests:** `test_stochastic_condition_encoder_learns` (VAE encoder has the var head + still learns the
  conditional translation) and `_reproducible` (reparam path bit-reproducible).

**Not yet / dropped:** **k-samples-per-source — dropped** (verified *not* implemented in cellflow: its GENOT
draws one latent + one target per source; it'd be ahead of the reference and nothing needs it). Classifier-
free guidance — deferred.

---

## ✅ Persistence shipped (`FlowMatching.save()`/`load()`) **[P0.2]**

`save(path)` writes `path/weights.pt` (torch `state_dict`) + `path/state.pkl` (cloudpickle — ctor config,
`spec`, the compiled `CompiledDims`, and the fitted `condition_fn` closure; same cloudpickle-of-a-closure
pattern already used by `sc_flow.external`). `load(path)` rebuilds the VF from the persisted dims + config,
loads the weights, and restores `condition_fn` — `predict()` works immediately after reload (does **not**
restore optimizer/trainer state, so `fit()` after `load()` starts fresh, not resumed). `cloudpickle` is
already a project dependency; `condition_fn`/`CompiledDims`/`FlowSpec` all roundtrip cleanly (verified).
`_build_vf` was refactored to take `CompiledDims` directly (was `CompiledData`) so `load()` doesn't need a
fake compile step. Tests: `test_save_load_roundtrip[otfm|genot]` (predict after reload is bit-identical to
before saving) + `test_save_before_fit_raises`. cf-train can use `(path/"weights.pt").exists()` for its
resume/skip check. Also added `scripts/smoke_train.py` — a standalone (non-pytest) fit+predict script with
a real learning assertion, used to smoke-test the stack on a fresh machine (e.g. a cluster GPU node).

## ✅ GPU-native OT coupling (data stays on the GPU)

The batch lives where the model lives (Lightning moves it to CUDA). The coupling now keeps the reps **and**
the transport plan on that device — `backends/jax/coupling/_device.py::couple_device` does torch tensor →
JAX array via **zero-copy DLPack** (same device) → ott sinkhorn / GW → `sample_joint` → indices back to
torch via DLPack. Only the tiny integer index arrays are produced; no cell data or coupling matrix is
copied to host. The `_TorchOTObjective` was rewritten to normalize the batch to torch tensors on the model
device and reorder there; the OT plan-sampling is seeded by a JAX `PRNGKey` (deterministic → bit-repro
holds on CPU). `_device.py` sets `XLA_PYTHON_CLIENT_PREALLOCATE=false` (setdefault) so jax — which only does
the small coupling — doesn't grab ~75% of VRAM and starve the torch model. `couple_device` is validated on
CPU (DLPack CPU) and on an **A100** (probe + `smoke_train.py --device cuda` for otfm **and** genot: both
train end-to-end and learn the conditional shift, reps/plan confirmed on `cuda:0`).

**Perf note:** on the *toy* model, GPU is slower than CPU (~0.65 s/step) — per-step torch↔jax handoff +
jax dispatch overhead dominates trivial compute. Amortizes on a real large model, but the per-step coupling
overhead is a profiling item for the Tahoe run (`ml-performance-audit`).

### GPU build recipe (both torch and jax on CUDA, matched to the node's driver)

Both must be CUDA builds. The node's driver caps the CUDA version (e.g. driver 575 → CUDA 12.9), and
`torch cu130` needs driver 580+, so **don't** hard-pin a CUDA torch. Instead:

- **torch**: `UV_TORCH_BACKEND=auto` (env var — the `--torch-backend` *flag* only works with `uv pip`, not
  `uv sync`) auto-detects the driver and picks the matching wheel (e.g. cu129). Re-sync **won't** swap an
  already-installed torch — force it: `uv pip install --torch-backend=auto --reinstall-package torch torch`.
- **jax**: the `cuda` extra (`pyproject.toml`) adds `jax[cuda12]` (Linux only) — self-contained CUDA-12
  wheels from PyPI. So on a GPU node: `UV_TORCH_BACKEND=auto uv sync --extra all --extra cuda --extra test`.
- Interactive session used: lab **dropbear** (`submit_dropbear.sbatch` on `gpu_p`/`gpu_priority` for a
  driver-580 / newer-driver node; ssh `node-<session>` with `StrictHostKeyChecking=no`). The dropbear shell
  doesn't inherit SLURM's GPU env — export `CUDA_VISIBLE_DEVICES=0` in commands. (`interactive_gpu` gave an
  older-driver V100S; `gpu_priority` gave A100/H100.)

## Next up (todo)

Ordered for a faithful cf-train Tahoe run (the ⏭ rows of `cellflow-vendor-fm.md` §8), then model polish:

1. **Validation loop + metrics** — a `validation_step` in the harness over a held-out split; add `r_squared`
   (per-condition predicted-vs-target-mean R²) next to the existing `EnergyDistance`; expose a
   `metrics_history` on the model and emit `val_<metric>_mean` for the sweep objective. **[P0.3/P1.2/P1.3]**
2. **Held-out split** — `split_by` + `split_ratios` over the compiled conditions (deterministic by seed) →
   train/val loaders, surfaced through `FlowMatching.fit`. **[P1.1]**
3. **Logger passthrough** — a `logger=` param on `fit` (None / wandb). **[P2.1]**
4. **Classifier-free guidance** (GENOT + OTFM) — condition-dropout in training + a guided `predict`. **[G3]**

`uv.lock` is currently untracked (generated by the 3.12 `uv sync`); decide whether to commit it (the repo
had no lockfile before).

## ✅ G1 shipped (GENOT-L, linear, same-space)

`FlowMatching(objective="genot")` trains + predicts end-to-end; `tests/test_flow_matching.py::test_genot_fit_and_predict`
passes (same-space + coupling-rep), full suite green (56 passed, 1 skipped). What landed:

- **`TorchGENOTObjective`** (`training/_objective.py`, registered `"genot"`): OT-resample `(source,target)`,
  sample target-space latent `~N(0,I)`, flow **latent→target** (`compute_xt(t, latent, target)`,
  `u = target-latent`), VF conditioned on the source cell (`model(t, x_t, cond, source=x0)`). Refactored the
  shared OT/condition/loss plumbing into module helpers + a `_TorchOTObjective` base that OTFM and GENOT share.
- **Source-encoder VF** — `FlowMatching._build_vf` enables `source_encoder_mlp_kwargs` (sized by the **state**
  dim; G1 is same-space, coupling reps only drive the OT plan) when `objective="genot"`.
- **Generative predict** — `FlowMatching.predict` branches: GENOT integrates from latent noise with `x` held
  as the source condition (noise→target|source); `predict(seed=...)` makes the (stochastic) sample
  reproducible. OTFM predict unchanged.
- **Seeding** — latent noise gets its own seeded CPU `torch.Generator` (seed+1), so GENOT training + predict
  are bit-reproducible like OTFM.

**Not yet (G2/G3):** quadratic/fused GW coupling (cross-space, `ot_quadratic_coupling`), k-samples-per-source,
classifier-free guidance, stochastic condition encoder.

---

## 0. What GENOT is, and why the port is small

**OTFM** (shipped): a *deterministic* flow **source → target**, VF conditioned on the perturbation. The OT
plan pairs `(source, target)`; the flow starts at the source cell.

**GENOT** ([Klein et al. '23], cellflow `solvers/_genot.py`): a *generative* flow **noise → target**, the VF
conditioned on the **source cell `x₀`** (plus the perturbation). The OT plan still pairs `(source, target)`,
but the source **only conditions** the VF — the flow itself goes from a latent noise sample to the target.
Concretely (cellflow `_genot.py` loss, verbatim intent):

```
latent = latent_noise_fn(rng, (n, target_dim))          # ~ N(0, I) in target space
x_t    = probability_path.compute_xt(t, latent, target) # flow LATENT → target  (not source → target)
u_t    = probability_path.compute_ut(t, x_t, latent, target)   # = target - latent
v_t    = vf(t, x_t, cond=perturbation, source=x₀)       # source CONDITIONS the field
loss   = mean((v_t - u_t)²)  [+ encoder reg]
```

**Why small:** the torch `MLPVelocity` **already supports source conditioning** — `forward(t, x,
condition_dict, source)`, `source_encoder_mlp_kwargs`/`use_source_encoder`, `_get_encoded_source`, and
`_conditioning_dim` already folds the source-encoder output into the conditioning vector
([`nn/_vf.py`](../../src/sc_flow/backends/torch/nn/_vf.py)). And the JAX quadratic coupling already exists —
`ot_quadratic_coupling` (GW/fused) in
[`backends/jax/coupling/_coupling.py`](../../src/sc_flow/backends/jax/coupling/_coupling.py). So GENOT is
**a new objective + a predict branch + wiring**, not new architecture or new numerics.

---

## 1. Features to add (the list)

| # | Feature | Where | New? |
|---|---------|-------|------|
| F1 | **Latent noise sampler** — target-space `N(0, I)`, seeded (`torch.randn(n, target_dim, generator=...)`) | objective helper | new (tiny) |
| F2 | **`TorchGENOTObjective`** (register `"genot"`) — OT-match → sample latent → flow **latent→target**, VF conditioned on `source` cell; loss + encoder reg | `training/_objective.py` | new |
| F3 | **VF with source encoder** — build `MLPVelocity` with `source_encoder_mlp_kwargs` sized by the source-rep dim | `FlowMatching._build_vf` | wire existing |
| F4 | **Source dim** — expose the source-rep feature dim for F3 | `CompiledDims` (derive from `coupling` src rep, else state) | tiny |
| F5 | **GENOT predict** — sample latent, integrate **from noise** conditioned on the source cell `x` (OTFM integrates `x` itself) | `FlowMatching.predict` | new branch |
| F6 | **Quadratic/GW coupling (GENOT-Q)** — route `is_quadratic` coupling schemas to `ot_quadratic_coupling`; matching data = `src_quad`/`tgt_quad` (+ `lin` for fused) | objective `_match_reps` | wire existing |
| F7 | **Objective selection** — `objective="genot"` selects F2 + F3-VF + F5-predict; `otfm` unchanged (both resolved by name via `build_objective`) | `FlowMatching` | wire |

Deferred GENOT options (not in this plan): k-samples-per-source (multiple latents per pair), classifier-free
guidance, `latent_noise_fn` overrides beyond standard normal.

---

## 2. The differences that actually matter (OTFM → GENOT)

Three deltas, everything else is shared with the OTFM path:

1. **Flow endpoints.** OTFM: `x_t = path.compute_xt(t, source, target)`, `u = target - source`.
   GENOT: `x_t = path.compute_xt(t, latent, target)`, `u = target - latent`, with `latent` sampled fresh
   each step. → the source no longer enters the *flow*.
2. **Source as a conditioning input.** GENOT passes the (OT-resampled) source cell into the VF's `source=`
   arg; OTFM passes `source=None`. → the VF must be built **with** the source encoder for GENOT.
3. **Predict.** OTFM integrates the ODE from the real cell `x` (`y0 = x`, `source=None`). GENOT integrates
   from noise (`y0 = latent`) with `source = x` (the cell you're translating) held fixed across the
   integration. → GENOT predict is *stochastic* (depends on the noise draw / a predict seed).

The OT coupling step (resample `(source, target)` by the plan), the probability path, the torch loss shape,
the Lightning harness, `CompiledData.dims`, and bit-repro seeding are all **reused unchanged**.

---

## 3. Plan (phased)

**Phase G1 — GENOT-L (linear, same/aligned space): the usable core.**
- F1 latent sampler (seeded off the objective's `torch.Generator`).
- F2 `TorchGENOTObjective` using linear `ot_linear_coupling` (reuse `_match_reps`), flow latent→target,
  source-conditioned VF, encoder reg (same as OTFM).
- F3/F4: `_build_vf` enables `source_encoder_mlp_kwargs` when `objective="genot"`; source dim =
  `dims.coupling["src_lin"]` if a coupling schema is set, else `dims.state` (same-space).
- F5 GENOT predict branch (sample latent, `y0 = latent`, `source = x`), with a `predict(seed=...)` / rng for
  the noise so it's reproducible.
- F7 selection; `otfm` path untouched.
- **Test:** `test_genot_fit_and_predict` (shape + finite + bit-repro like OTFM), on the toy adata with a
  coupling schema; assert the VF has a source encoder and predict is deterministic given a seed.

**Phase G2 — GENOT-Q (quadratic / fused GW, cross-space).**
- F6: `_match_reps` (or a GENOT-specific `_prepare_data`) detects `CouplingDataSchema.is_quadratic` and calls
  `ot_quadratic_coupling(src_quad, tgt_quad[, src_lin, tgt_lin])`; the source fed to the VF is `src_lin`
  (fused) or `src_quad`. Cross-space: state_dim (x_t/target) may differ from source_dim.
- binded already streams `src_quad`/`tgt_quad` when the coupling schema names them (compile_obs coupling
  locs). Confirm the batch carries them; extend `CompiledDims.coupling` already covers quad dims.
- **Test:** a quadratic coupling schema (`src_quad`/`tgt_quad`) fits + predicts.

**Phase G3 — options (deferred):** k-samples-per-source, CFG guidance.

---

## 4. Sketch (Phase G1 objective)

```python
@register_objective("genot")
class TorchGENOTObjective(Objective):
    # __init__ like TorchOTFMObjective + a seeded latent generator (target-space N(0,I)).
    def compute_loss(self, model, batch):
        src_state, tgt_state = np(batch["source"]), np(batch["target"])
        src_rep, tgt_rep = self._match_reps(batch)                 # reused
        src_ixs, tgt_ixs = ot_linear_coupling(src_rep, tgt_rep, rng=self._np_rng, ...)  # reused
        x0_source = torch(src_state[src_ixs])                      # the SOURCE cell (conditions the VF)
        target    = torch(tgt_state[tgt_ixs])
        latent    = torch.randn(target.shape, generator=self._latent_gen)  # F1
        t   = torch.rand(n,1, generator=self._t_gen)
        x_t = self._path.compute_xt(t, latent, target)            # flow noise -> target
        u   = self._path.compute_ut(t, x_t, latent, target)       # target - latent
        v   = model(t, x_t, cond_t, source=x0_source)             # F2/F3: source-conditioned
        loss = ((v - u)**2).mean() + encoder_reg(model, cond_t)   # reused reg
        return loss, {...}
```

`predict` (F5): `latent = randn(x.shape, gen)`, integrate `f(t,y)=model(t_exp, y, cond, source=x)` from
`y0 = latent`; OTFM keeps `y0 = x, source=None`. Branch on `self.objective_name`.

---

## 5. Open decisions (Selman)

- **Source rep for GENOT-L with no coupling schema:** default to the **state rep** (same-space, source=x in
  X/X_pca) — OK? Or require a `CouplingDataSchema` for GENOT so the source space is always explicit?
  → **Selman:**

- **Predict stochasticity:** expose `predict(seed=...)` (or an `n_samples` to average) since GENOT predict
  samples noise — one sample by default, seeded off the model seed? → **Selman:**

- **One `"genot"` objective handling lin+quad** (branch on `is_quadratic`) vs separate
  `genot-l`/`genot-q`?
  → **Selman:**

- **Scope now:** G1 only (linear — covers cf-train same-space genot), or G1+G2 (add quadratic/GW)?
  → **Selman:**

- **State vs source dim when cross-space (G2):** `state_dim` sizes `x_t`/decoder (target space);
  `source_encoder` sized by source dim. Confirm cf-train's genot is same-space (so this is moot for G1).
  → **Selman:**

- **Anything else:** → **Selman:**
