# Plan: torch-native OTFM, with OT coupling kept in JAX

**Branch:** `feat/refactors` · **Status:** proposal (v2 — pivoted to torch-native) · **Decisions:** see "Selman's notes".

> **v2 supersedes v1.** v1 proposed vendoring CellFlow's flax velocity field and keeping the JAX↔torch
> bridge. Two things killed that: (a) the current path isn't even faithful OTFM (the minibatch OT coupling
> is missing — see §1), and (b) once you have to add OT anyway, keeping a JAX VF is the *only* thing the
> bridge still buys. Decision: **port VF/encoder/path/loss to torch; keep only the OT coupling
> (`match_linear` + `sample_joint`) in JAX for now; delete the bridge.**

## ✅ Implemented (this session — the five core rows)

Torch-native OTFM `fit`/`predict` runs end-to-end with **no cellflow and no bridge**; `tests/test_flow_matching.py`
passes (base + coupling-rep×{sinkhorn, independent}); full configured suite green (53 passed, 1 skipped).

- **OT kernel** — reused the existing `backends/jax/coupling.ot_linear_coupling` (ott sinkhorn + plan-sampling)
  and `independent_coupling`; no new `_match.py` needed. It's the one JAX call, forward-only.
- **`CompiledData.dims`** (`data/_compile_obs.py`) — `CompiledDims(state, condition{realm:dim}, max_comb,
  coupling{role:dim})` from `key_backings(...).shape` + one `condition_fn` lookup (no sampler).
- **Torch VF** — `MLPVelocity` sized from `compiled.dims` in `FlowMatching._build_vf`; removed the dead
  `init_from_dims_registry` + `sc_flow.data._dims_registry` import from `nn/_vf.py` and `nn/_modules.py`.
- **Torch OT-FM objective** — `TorchOTFMObjective` (`training/_objective.py`, registered `"otfm"`;
  `FlowMatching.fit` selects the objective by name via `build_objective`): OT-reorder
  → straight-path CFM loss (torch probability path) → deterministic encoder reg (`0.5·mean(emb²)`, via new
  `MLPVelocity.encode_condition`). Runs on the shared `SCFlowLightningModule`.
- **`FlowMatching`** (`_model.py`) — torch-native `fit` (compile→dims→VF→objective→Lightning) and `predict`
  (torchdiffeq); knobs `hidden_dims`/`decoder_dims`/`time_encoder_dims`/`pooling`/`condition_mode`/
  `regularization`/`sigma`/`match_method` forwarded natively.
- **Bit-reproducibility** — a single `FlowMatching(seed=...)` seeds every stochastic source: VF init
  (`torch.manual_seed`), binded data order (`Scheme` seed, threaded through `compile_obs`/`FlowSpec.compile`/
  `build_loader`), OT plan-sampling (explicit `np.random.Generator` on `ot_linear_coupling`/
  `independent_coupling`/`_select_indices` — no more process-global `np.random`), the per-step `t` draw
  (CPU `torch.Generator`), and probability-path noise. `test_flow_matching_bit_reproducible` asserts
  same-seed → bit-identical weights + predictions, different-seed → divergent. (Sinkhorn itself is a
  deterministic solve.)
- **Bridge deleted** — removed `backends/torch/jaxbridge/` + its tests; pruned the jax-objective harness test.
- **Env/packaging** — floor bumped to **Python ≥3.12** (binded requires it; `requires-python`, classifiers,
  `.python-version`, hatch matrix); `all`/`test-torch` extras now include `lightning` (+`jax` for the OT kernel).

**Deliberately NOT done** (deferred, unchanged from §8): legacy-move of `backends/torch/methods/` (dead, off
the live path — the torch backend `__init__` is lazy so it never loads); save/load; split; `validation_step` +
r_squared/e_distance; wandb logger; GENOT; stochastic condition encoder; attention pooling.

---

## 0. The target, in one paragraph

`FlowMatching.fit` trains a **torch** velocity field with a **torch** FM loss via `loss.backward()` and
Lightning. Each minibatch, the **only** JAX call is a forward-only OT step: feed the source/target coupling
reps into `match_linear` (ott sinkhorn) → transport plan `tmat` → `sample_joint` → integer indices → reorder
`(source, target)` before the torch loss. No gradient crosses into JAX, so the DLPack autograd bridge is
**deleted**. `predict` integrates the torch VF with `torchdiffeq` (already implemented). JAX shrinks from
`{jax, flax, optax, ott-jax, diffrax}` to just `{jax, ott-jax}`, and only for OT.

Framework split:

| Concern | Framework | Where |
|---|---|---|
| Velocity field, set/condition encoder, conditioning | **torch** | `backends/torch/nn/` (`_vf.py`, `_set_encoder.py`, `_conditioning_layers.py`, `_time_features.py`) — exists, rewire |
| Probability path (`ConstantNoiseFlow`) | **torch** | `backends/torch/probability_paths/` — exists |
| FM + encoder loss | **torch** | `loss.backward()`; ~6 lines |
| Training loop | **torch/Lightning** | `backends/torch/training/_harness.py` — rewire |
| **OT coupling** (`match_linear` + `sample_joint`) | **JAX** (for now) | new small kernel, forward-only, no bridge |
| ODE (predict only) | **torch** | `torchdiffeq` in `_model.py:200` — already there; FM training has no ODE |
| GENOT | **JAX** (deferred) | `backends/jax/` — untouched, port later |

---

## 1. Why v1 was wrong: the current path is vanilla CFM, not OTFM

cellflow's OT lives in `step_fn`, **not** `loss_fn` ([cellflow `_otfm.py:241-246`]): `tmat =
match_fn(src, tgt)` → `sample_joint(rng, tmat)` → `src, tgt = src[src_ixs], tgt[tgt_ixs]`, *then* the
identical 4-line loss. `match_fn=None` is cellflow's explicit vanilla-CFM mode.

This repo vendored **only `loss_fn`** and (per its own `jaxbridge/README.md`) assumed matching happens
"upstream." But nothing upstream does OT:
- binded pairs source↔target by **matched context/label** (`BoundClassSampler`, common columns) — **no
  sinkhorn** anywhere in binded.
- the adapter ([_adapter.py:41-42](src/sc_flow/backends/torch/jaxbridge/_adapter.py)) takes `source`/`target`
  and **discards `source_reps`/`target_reps`** — the exact reps `CouplingDataSchema` streams to feed OT.

So today's `FlowMatching` runs the `match_fn=None` regime — **independent-coupling CFM**. For cf-train's
`solver: otfm`, the minibatch OT reorder is **net-new work** (it was never wired, on any path). The pieces
exist as intent — `CouplingDataSchema` (src_lin/tgt_lin/…), streamed coupling reps, and the dead torch
`match_fn`/`LinCouplingMethod` seam — but the wire from reps → `tmat` → resample → loss must be built.

**Decision (confirmed):** the Tahoe run needs **true minibatch-OT**. OT + `sample_joint` stay in **JAX** for
now (proven `ott` sinkhorn); migrate to torch `pot` later.

---

## 2. What already exists in torch (reuse — do not rewrite)

`backends/torch/__init__.py:5` exports `coupling, metrics, methods, nn, probability_paths, solvers`. Present:

- **VF:** `nn/_vf.py` `MLPVelocity` — time/state/condition/source encoders, `condition_encoder_pooling_mode ∈
  {"mean","sum"}`, conditioning via `_conditioning_layers.py`, time features via `_time_features.py`.
- **Set encoder:** `nn/_set_encoder.py`. **Probability paths:** `probability_paths/_probability_paths.py`.
- **Solvers:** `solvers/_ode_solver.py` / `_sde_solver.py`. **Coupling:** `coupling/_coupling.py` (pot — the
  eventual home when OT moves off JAX). **Metrics:** `metrics/_metrics.py` (has `EnergyDistance`).
- **Training:** `training/_harness.py` (`SCFlowLightningModule`), `training/_objective.py`.

It's dead **only** because it's wired to the removed in-memory data layer
(`DistributionData`/`MatchedDistributions`/`CouplingData`/`DataDimensionalitiesRegistry`), not because it's
incomplete and not because it needs cellflow. The work is **rewire onto binded batches + `CompiledData.dims`**.

**Net-new torch (only as configs demand):** attention pooling (`token`/`seed` — `MLPVelocity` is mean/sum
only), stochastic/VAE condition encoder + KL, classifier-free-guidance dropout, FiLM/ResNet if
`_conditioning_layers.py` lacks them. **cf-train's Tahoe config (mean pooling, deterministic, concatenation)
needs none of these.**

---

## 3. The JAX OT kernel (the only JAX that remains)

New module, e.g. `src/sc_flow/backends/jax/coupling/_match.py` (or fold into existing `backends/jax`):

```python
# forward-only; no autograd, no DLPack bridge. jax arrays in, index arrays out.
def match_linear(src_rep, tgt_rep, *, epsilon, tau_a=1.0, tau_b=1.0, **sinkhorn_kwargs) -> tmat: ...
def sample_joint(rng, tmat) -> (src_ixs, tgt_ixs): ...
```
- `match_linear` = `ott` sinkhorn (`ott.geometry.pointcloud` + `ott.solvers.linear.sinkhorn`), copied/adapted
  from cellflow `utils.py:14`. `sample_joint` = sample paired indices from the plan (from `ott`/cellflow
  solver utils; ~10 lines of `jax.random.choice` over the flattened plan). Vendor or reimplement both — tiny,
  MIT/Apache, self-contained (`jax` + `ott` only).
- **Called in the torch training step** (mirrors cellflow `step_fn`): given a binded batch with `source`,
  `target`, and coupling reps `source_reps`/`target_reps`, compute indices in JAX, bring them back as numpy
  (`np.asarray` — indices are small, no DLPack needed), reorder the torch `source`/`target` (**and the
  per-cell `condition`** — see §7 risk) before the torch loss.
- **Quadratic/GW** (`CouplingDataSchema.is_quadratic`, for GENOT later): `ott` GW solver, same shape. Out of
  scope now.
- Migration path: swap this JAX kernel for torch `coupling/_coupling.py` (pot) when you're ready — the
  training-step seam (a `match_fn(src_rep, tgt_rep) -> indices` callable) stays identical.

---

## 4. VF ↔ ODE are coupled; OT is not

- **OT + `sample_joint`** are separable — forward-only, operate on reps, emit indices. Stay JAX trivially. ✅
- **VF + ODE** must share a framework — the ODE integrates the VF; a JAX/diffrax solve around a torch VF
  can't be `jit`-traced (torch ops inside jax) and a per-step reverse bridge kills vectorization. Since the
  VF is torch, **predict's ODE is `torchdiffeq`** — which `_model.py:200` already does. FM *training* has **no
  ODE** (closed-form `u_t`). So "keep ODE in JAX for now" is effectively already satisfied: nothing in the FM
  path needs a JAX ODE. Your "confirm torchdiffeq is vectorized" concern is real but torch-side and
  non-blocking — verify it **condition-batches** like cellflow's `jit(vmap)` diffrax predict
  (cellflow `_otfm.py` `_get_flat_predict_fn`); if not, batch the ODE over conditions in `predict`.

---

## 5. File layout: add / rewire / delete

**Add**
```
src/sc_flow/backends/jax/coupling/_match.py     # match_linear + sample_joint (JAX OT kernel)
src/sc_flow/backends/jax/coupling/__init__.py
+ LICENSE / PROVENANCE notes for the ott-derived bits (Apache-2.0)
```

**Rewire (existing torch stack → binded batch + CompiledData.dims)**
- `nn/_vf.py`: `MLPVelocity.__init__` fed from `CompiledData.dims` instead of `init_from_dims_registry(...)`
  (the plain `__init__` already exists; the classmethod is just an unpacker). Ensure `forward` returns what
  the loss needs (`v_t`, plus condition `mean`/`logvar` for the encoder reg — see §7).
- `training/_harness.py`: a `SCFlowLightningModule.training_step` that (1) OT-reorders the batch via §3,
  (2) samples `t`, (3) `x_t = path.compute_xt(t, src, tgt)`, (4) `v_t = vf(t, x_t, cond)`, (5) `loss =
  mse(v_t, tgt - src) + encoder_reg`, (6) `return loss`. Add `validation_step` here later (cf-train P1.2).
- `_model.py` `FlowMatching.fit`: build the torch VF from `compiled.dims`, wrap in the torch harness, run
  `pl.Trainer`. `predict`: torch VF + `torchdiffeq` (drop the JAX apply closure at `_model.py:180-197`).
- Revive the `match_fn` seam so the method/harness calls the §3 kernel each step (the dead
  `methods/_base.py` `_call_match_fn_safe` is the design reference — reimplement minimally against the binded
  batch; don't resurrect its `DistributionData` plumbing).

**Delete (once VF/loss are torch)**
```
src/sc_flow/backends/torch/jaxbridge/          # whole package: _bridge, _cellflow, _objective, _adapter
```
`JaxParamModule`, `JaxLossFunction`, `CellFlowFMObjective`, `make_fm_value_and_grad`, `iter_fm_batches` — all
gone. Keep the `time`/`encoder_noise` sampling logic from `_adapter.py` (move it into the torch step).

**Legacy quarantine** (unchanged from v1): move `backends/torch/methods/` →
`src/sc_flow/legacy/torch_methods/` (dead: imports removed data types); repoint the two lazy importers
(`trainer/_lightning.py:78`, `methods/_custom.py:19-20`). Quarantine, don't repair. Decide whether the
`nn/_vf.py` **dims-registry coupling** is removed in place (recommended, since we're reviving `MLPVelocity`)
vs. moved.

---

## 6. Config knobs & dims (torch VF constructor)

**Knobs → `MLPVelocity.__init__` args** (torch-native, so no forwarding to a flax class):
cf-train `hidden_dims`/`decoder_dims`/`time_encoder_dims`/`condition_embedding_dim` → the corresponding
`MLPVelocity` mlp-kwargs / output-dims; `pooling: mean` → `condition_encoder_pooling_mode="mean"` (already
supported); `condition_mode`/`regularization` → the encoder-reg branch of the loss. Map cf-train's names to
`MLPVelocity`'s (`state_encoder_mlp_kwargs`, `time_encoder_*`, `condition_encoder_*`, `vf_decoder_mlp_kwargs`)
in `FlowMatching.__init__`. **P2.2 becomes native**, no flax indirection.

**Dims without the sampler** (unchanged from v1): add `dims` to `CompiledData` ([_compile_obs.py:225](src/sc_flow/data/_compile_obs.py))
from `binded._io.key_backings(src, "obsm/"+sample_rep or "X")[0].shape[1]` (header read, no cells);
`fit` sizes the VF from `compiled.dims["state"]` (+ coupling lin/quad dims from the coupling locs) instead of
pulling a batch. Condition input dims: the torch encoder infers them from the first batch's condition arrays,
or from `compiled.dims` — prefer `compiled.dims` so construction stays batch-free.

---

## 7. Parity & wiring risks (verify while implementing)

1. **Encoder regularization return.** The FM loss adds `0.5·mean(cond_mean²)` (deterministic, reg>0) and a KL
   term (stochastic). The torch VF/encoder must expose the pooled condition **mean** (and **logvar** for
   stochastic) — confirm `nn/_set_encoder.py` returns them; add if missing. For cf-train (deterministic,
   reg=1.0) only the mean penalty is needed.
2. **OT reorder must keep condition aligned.** After `src, tgt = src[src_ixs], tgt[tgt_ixs]`, the **per-cell
   condition** must be reordered by `tgt_ixs` (condition belongs to the target/perturbed cell). Confirm
   whether binded's batch condition is per-cell or per-batch; if a batch mixes conditions (root draws a class
   schedule ∝ weights), reorder condition with the target. Getting this wrong silently trains mismatched
   (cell, condition) pairs.
3. **Match on coupling reps, not state.** OT runs on `source_reps[src_lin]`/`target_reps[tgt_lin]` (which may
   differ from the state rep), but the reorder applies to the **state** source/target. Keep rep↔state row
   alignment (binded streams them aligned).
4. **sinkhorn epsilon/scaling on 100M.** `match_linear`'s epsilon/threshold and low-rank vs full sinkhorn
   matter for a 100M run's per-batch cost and numerical stability — expose them as `fit`/config knobs; don't
   hardcode.
5. **torchdiffeq vectorization** (§4) — condition-batch predict if needed.
6. **Numerical parity** vs cellflow won't be bit-exact; any existing flax checkpoint is unloadable (fine for
   fresh training). Add a small train-a-few-steps + predict smoke test as the acceptance bar.

---

## 8. cf-train coverage map (v2)

| cf-train item | v2 approach | Scope |
|---|---|---|
| **P0.1** cellflow dep | gone — nothing imports cellflow; VF is torch, OT kernel is vendored `ott` | ✅ core |
| **OTFM faithfulness** | §3 JAX OT kernel wired into the torch step (was missing) | ✅ core (net-new) |
| **P2.2** decoder/time/pooling | native `MLPVelocity` args (§6) | ✅ core |
| dims without sampler | `CompiledData.dims` (§6) | ✅ core |
| torch-native layer → legacy | §5 | ✅ core |
| **P0.2** save/load | torch `state_dict` (VF) + pickled spec + OT-kernel config + `_condition_fn` | ⏭ follow-on |
| **P0.3 / P1.2 / P1.3** val logs / `validation_step` / `r_squared`+`e_distance` | `validation_step` in `_harness.py`; `EnergyDistance` exists, add `r_squared`; emit `val_<metric>_mean` | ⏭ follow-on |
| **P1.1** split_by/split_ratios | deterministic split of compiled conditions → val loader | ⏭ follow-on |
| **P2.1** wandb logger | `logger=` param on `fit` | ⏭ follow-on |
| **P2.3** genot | port GENOT (JAX today, `backends/jax`); quadratic OT via `ott` GW; the big one | ⏭ deferred |
| **P3.1 / P3.2** sample_rep_dims / min_runs_per_leaf | pre-slice cf-train-side; document `min_runs_per_leaf` semantics | ⏭ minor |

**This plan's core = the five ✅ rows.** The ⏭ rows are framework-agnostic and land after.

---

## 9. Effort

- **Tier 1 — rewire the torch stack onto binded + wire OT (the bulk).** `MLPVelocity` off dims-registry;
  torch `training_step` with the OT reorder + closed-form FM loss; `fit`/`predict` repointed; bridge deleted;
  legacy moved. cf-train's config needs no new NN modules. Estimate: several focused days — most of it is the
  data-seam rewire already on your worklist, plus the (small) OT kernel and its wiring/alignment.
- **Tier 2 — parity modules, lazy:** attention pooling, stochastic encoder + KL, CFG, FiLM/ResNet — only when
  a config needs them.
- **Deferred — GENOT port.**

---

## 10. Task description for the next agent (self-contained)

> **Goal:** Make `sc_flow.FlowMatching` a **torch-native OTFM**: torch velocity field + torch FM loss trained
> via Lightning, with the **minibatch OT coupling kept in JAX** (`ott` sinkhorn `match_linear` +
> `sample_joint`) as a forward-only batch-reorder. Delete the JAX↔torch autograd bridge. No `cellflow`
> import anywhere.
>
> **Do, in order:**
> 1. **JAX OT kernel** (§3): add `backends/jax/coupling/_match.py` with `match_linear` (adapt cellflow
>    `utils.py:14`, `ott` sinkhorn) and `sample_joint` (index-sample from the plan). Forward-only; jax in,
>    numpy indices out. Add ott (Apache-2.0) provenance. Expose epsilon/tau as params.
> 2. **Dims** (§6): add `dims` to `CompiledData` via `binded._io.key_backings(src, state_loc)[0].shape[1]` at
>    `_compile_obs.py:225` (+ coupling lin/quad dims).
> 3. **Torch VF** (§5/§6): give `MLPVelocity` a `from_dims(dims, **knobs)` construction path off
>    `init_from_dims_registry`; forward cf-train knobs (hidden/decoder/time dims, pooling, condition_mode,
>    regularization) as native args; ensure `forward` returns `v_t` + condition `mean`/`logvar`.
> 4. **Torch training step** (§5): rewrite `SCFlowLightningModule.training_step` — OT-reorder the batch via
>    (1) (reorder condition with the target, §7.2), sample `t`, `x_t = path.compute_xt(...)`, `v_t = vf(...)`,
>    `loss = mse(v_t, tgt - src) + encoder_reg`, `loss.backward()` (Lightning). Reuse torch
>    `probability_paths` + `metrics`.
> 5. **`fit`/`predict`** (`_model.py`): build torch VF from `compiled.dims`, run the torch harness; `predict`
>    = torch VF + `torchdiffeq` (drop the JAX apply closure). Condition-batch the ODE if needed (§4).
> 6. **Delete** `backends/torch/jaxbridge/` entirely; keep only the `time`/`encoder_noise` sampling (move into
>    the step). **Legacy:** move `backends/torch/methods/` → `legacy/torch_methods/`, repoint the two lazy
>    importers, remove the `nn/_vf.py` dims-registry coupling in place.
> 7. **Extras** (§11): shrink JAX to `{jax, ott-jax}`; drop flax/optax/diffrax from what `fit` needs; ensure
>    `torch` extra (has `pot`, `torchdiffeq`) + the OT extra cover a working `fit`.
>
> **Verify:** clean env; `grep -rn "cellflow\|jaxbridge\|JaxParamModule" src/` is empty (docstrings aside);
> `python -c "import sc_flow.data"` needs no jax; a tiny `fit(...).predict(...)` round-trips on CPU **with OT
> matching on** (`match_fn` firing — assert the reorder changes pairing) and `pooling="mean"`,
> deterministic. Check the encoder-reg term is non-trivial. Update/replace `tests/test_flow_matching.py`
> (currently exercises the bridge) and delete `tests/backends/torch/jaxbridge/`.
>
> **Out of scope:** save/load, split, validation_step + metrics, wandb, GENOT. Tier-2 parity modules only if
> a test needs them.

---

## 11. Extras (`pyproject.toml`)

JAX footprint shrinks — VF is torch (no flax, no optax), predict ODE is torch (no diffrax). Only OT needs JAX:
```toml
optional-dependencies.ot = [ "jax", "ott-jax" ]     # the minibatch OT kernel only
optional-dependencies.all = [ "sc-flow-tools[torch,lightning,ot]" ]
optional-dependencies.test-torch = [ "sc-flow-tools[torch,lightning,ot,test]" ]
```
`torch` already carries `pot` + `torchdiffeq`. Drop the old `jax` extra's `flax`/`diffrax` if nothing else
uses them (grep first — GENOT in `backends/jax` may). Naming: `ot` reads truer than `fm` now that the extra
is *only* the OT kernel — **your call** (you said `fm` earlier, but the surface changed).

---

## 12. Selman's notes / feedback (recommendation first)

- **ODE/VF coupling:** confirm you're OK that porting the VF to torch pulls predict's ODE to `torchdiffeq`
  now (already implemented); "JAX ODE" would require keeping a JAX VF too. → **Selman:**

- **OT wiring location:** in the torch **training step** (recommended, mirrors cellflow `step_fn`) vs. in a
  data-pipeline wrapper before the loader yields. → **Selman:**

- **Extra name:** `ot` (recommended, matches the shrunk surface) vs. keep `fm`. → **Selman:**

- **`nn/_vf.py` dims-registry coupling:** remove in place (recommended — we're reviving `MLPVelocity`) vs.
  move to legacy. → **Selman:**

- **`match_linear` deps:** vendor the ~2 functions from cellflow/ott (recommended) vs. depend on `ott-jax`
  and import its sinkhorn directly. → **Selman:**

- **Scope:** core-only (five ✅ rows) now vs. also start a ⏭ follow-on (e.g. validation_step). → **Selman:**

- **Anything else:** → **Selman:**
