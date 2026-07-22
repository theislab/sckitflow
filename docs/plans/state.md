# sc-flow-tools — State

> **Audience.** Contributors joining the shared single-cell ML training stack (flow-matching, pan-cellflow,
> inverse-problem, foundation-model fine-tuning, and the HPC pipeline). Read this before touching
> `sc-flow-tools`, `binded`, or `cf-train`. It captures *why* the stack is shaped the way it is, the
> current state, the extension points, and what you personally need to do to plug your model in.

---

## 0. TL;DR

We are consolidating several theislab ML efforts onto **one shared, layered, PyTorch + PyTorch-Lightning
base** so each project only writes its *model* (loss + architecture + inference), not the dataloading /
training / eval glue. Model weights are **always** a torch `nn.Module`; **JAX is isolated to the
optimal-transport coupling** (via zero-copy DLPack), never to the weights or the training loop.

The layers:

- **`binded`** — declarative, index-free adata streaming + split/coupling over `annbatch`.
- **`mlcore`** (today `sc_flow.core`, to be extracted) — the ML toolbox: the one Lightning `TrainingModule`
  + the `Objective` (loss) and `Predictor` (inference) seams + optimizer + generic `nn` + metrics.
- **`sc-flow-tools` / `sc_flow.flow`** — the flow-matching toolbox: velocity fields, probability paths,
  objectives, predict, and the JAX/OTT coupling bridge. Extends `mlcore`.
- **`cf-train`** — the HPC experiment/pipeline template (config + scripts = publication code).

---

## 1. Motivation (why this exists)

Everyone was re-implementing the same annbatch↔model glue and forking hacky dataloaders. `cellflow` is
capable but **JAX-only**, and its architecture/config is hard to extend and maintain. Meanwhile the
gold standard for training and *sharing* foundation models is **PyTorch / Hugging Face**.

So the plan: a common, maintainable, torch-first base that everyone extends, reviewed by each other, with
experiments living in pipeline repos (reproducible) rather than in packages maintained forever. After the
base exists, each person focuses on their model; a new loss or architecture is a small reviewed extension,
and pipeline steps (fast PCA, resharding, …) get shared instead of re-solved per person.

## 2. Non-negotiable principles

1. **Never depend on `cellflow`.** It is the *idea ancestor* we port from and credit — **not** a runtime
   dependency, ever. Anything currently routing through cellflow is a migration artifact to remove.
2. **Weights are always a torch `nn.Module`.** Optimization is always torch + Lightning.
3. **JAX only where it's strong (OT/OTT), isolated and lazy.** No JAX in the weights or the loop; the OT
   coupling is a per-minibatch call bridged by DLPack (no host copy). `import sc_flow.flow` pulls **no jax**.
4. **Lean dependencies — vendor small settled math, don't take heavy deps.** (See §9.)
5. **Reproducible pins.** Git dependencies are pinned to **immutable commits** (or tags), never mutable
   branches, so downstream `uv.lock` is deterministic.
6. **Commit style:** short, lowercase, 3–4 words, no author/co-author trailers.

## 3. Layered architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│ cf-train  (myapp)   experiment/pipeline template — HPC, config, scripts │  ← publication code
└───────────────▲─────────────────────────────────────────────────────┘
                │ depends on
┌───────────────┴─────────────────────────────────────────────────────┐
│ sc-flow-tools / sc_flow.flow    FLOW-MATCHING TOOLBOX                    │
│   velocity fields · probability paths · objectives (OTFM/GENOT/…)       │
│   ODE predict · JAX/OTT optimal-transport coupling bridge               │
└───────────────▲─────────────────────────────────────────────────────┘
                │ extends
┌───────────────┴─────────────────────────────────────────────────────┐
│ mlcore  (today sc_flow.core — to be extracted)   ML TOOLBOX             │
│   TrainingModule (Lightning) · Objective + Predictor seams · optimizer  │
│   generic nn backbones · metrics                                        │
└───────────────▲─────────────────────────────────────────────────────┘
                │ streams via
┌───────────────┴─────────────────────────────────────────────────────┐
│ binded          adata streaming + split/coupling over annbatch (fork)   │
└─────────────────────────────────────────────────────────────────────┘
```

**Why the `core` / `flow` split.** `core` is model-family-agnostic (it trains *any* model+objective) and
is meant to be extracted into a standalone `mlcore` package that a foundation-models toolbox could reuse as
a sibling of the flow-matching toolbox. `flow` is the flow-matching specialization. Keeping the seam clean
now makes the future extraction a mechanical `sc_flow.core → mlcore` lift.

## 4. Repos, dependency chain, pins

| Repo | Remote | Import name | Role |
|---|---|---|---|
| binded | `theislab/binded` | `binded` | data streaming/split/coupling over annbatch |
| annbatch (fork) | `selmanozleyen/annbatch` | `annbatch` | out-of-core zarr sampler; carries the bound-class sampler + a multi-dataset purity fix |
| sc-flow-tools | `theislab/sc-flow-tools` (fork: `selmanozleyen/…`) | `sc_flow` | ML toolbox (`core`) + flow-matching toolbox (`flow`) |
| cf-train | `theislab/cf-train` | `myapp` | HPC experiment template |

**Target dependency chain (what it MUST become):**

```
cf-train  ──►  sc-flow-tools  ──►  binded  ──►  annbatch (fork, immutable commit)
```

**Pins (immutable, by design):**
- `binded` → `annbatch @ git+https://github.com/selmanozleyen/annbatch.git@98e543a8a75…` (the multi-dataset
  purity fix; same bug class as scverse/annbatch#256/#257).
- `sc-flow-tools` → `binded @ git+https://github.com/theislab/binded@318dd6e42…` (immutable commit on
  `main`).
- `cf-train` → **currently** `cellflow-tools[annbatch] @ theislab/cellflow@feat/annbatch-loader` — **this is
  the artifact to delete** (see §11).

## 5. Current state (as of this handoff)

**binded — released-ready.** `main @ 318dd6e ("simplify doc")`, pushed. annbatch pinned to the immutable
fix commit. Provides `Loader` (training) and `EvalLoader` (control-rooted held-out eval). 55 tests green.

**sc-flow-tools — on `feat/refactors`, NOT yet pushed.** The refactor landed as a linear series of
small commits (messages, newest last):
`chore(deps): pin binded … drop ../binded override` → `quarantine unused subsystems` → `move data to core` →
`quarantine dead backends` → `flatten backends to core/flow` → `ruff cleanup` →
`add predictor protocol` → `add validation cap todo` → `add hub model sharing` →
`attribute ported components`.
(The first commit still carries its original conventional-commit message — a `3–4 word` reword was
attempted but rebased away; reword it on the force-push to match the commit style.)
Each is smoke-verified (`scripts/smoke_train.py`, OTFM, trains + predicts + learns the shift). **Publishing
needs a force-push** (the branch was reworded, so it diverged from the remote — the force-push is gated by a
safety classifier; a human runs it).

Layout:
```
sc_flow/
  core/   {data, nn, metrics, training}      ML toolbox (torch only, no jax)
  flow/   _vf, probability_paths/, _objectives, _predict, _set_encoder,
          _combiner, _time_features, coupling/ (jax-OT bridge)
  legacy/ config, dataset, external, methods, preprocessing, trainer,
          jax_*/ torch_*/                    quarantined; not on the train path
  _model.py   FlowMatching (facade wiring core + flow)
```

**cf-train — `main @ b3457b4`, merged & pushed.** Still depends on `cellflow-tools[annbatch]`; its code
still imports the *old* `dagloader` name (the loader was renamed to `binded`), and its `uv.lock`
re-resolution currently fails on a stale `binded @ …sc-flow-tools@…#subdirectory=../binded` (a relative-path
leak). All of this is resolved by the §11 migration, not by patching the cellflow chain.

## 6. The seams — how you extend the stack

The one generic Lightning module is **`sc_flow.core.training.TrainingModule`** (formerly
`SCFlowLightningModule`; renamed because it is *not* flow-matching-specific). It trains a `model` under an
`Objective`, optionally validates with a `Predictor` + metrics, and owns the optimizer. It knows nothing
about flow matching. You plug in via three registerable seams:

| Seam | Contract | Registry | Flow-matching impl |
|---|---|---|---|
| **`Objective`** | `compute_loss(model, batch) -> (loss, logs)` | `OBJECTIVE_REGISTRY` | `OTFMObjective`, `GENOTObjective`, `LinearFMObjective` (`@register_objective`) |
| **`Predictor`** | `predict(model, batch) -> pred` | `PREDICTOR_REGISTRY` | `ODEPredictor` (`@register_predictor("ode")`) |
| architecture | builder → `nn.Module` | `ARCHITECTURE_REGISTRY` | *(currently unused/dead — slated to drop; VF is built directly)* |

- **`Objective`** is the train-time seam (≈ HF `Trainer.compute_loss`). Loss math can run in torch or JAX;
  the module doesn't care.
- **`Predictor`** is the inference seam (≈ Lightning `predict_step`). Pure inference (returns `pred`); the
  module extracts `target`/identity-baseline and runs metrics. `FlowMatching.predict` and validation share
  the same `Predictor`, so a validation metric reflects exactly what inference does.

**Hub sharing (foundation-model story).** `core.nn.BaseModule` mixes in
`huggingface_hub.PyTorchModelHubMixin`, so **every** toolbox model gets `save_pretrained` /
`from_pretrained` / `push_to_hub` with **safetensors** + auto `config.json`, for free. We deliberately did
**not** adopt the HF `Trainer` (its supervised eval loop fights our population-level generative validation);
we took only the *sharing* win, additively. Weights round-trip verified.

## 7. What's unique here vs. generic (own vs. reuse)

Be honest about the moat so you reuse the settled parts and own the differentiators.

**Ours (own + iterate):**
- the perturbation **condition/set encoder** (`SetEncoder` — permutation-invariant Deep-Sets over a
  *set* of perturbation covariates; this is the modeling idea),
- context-matched **JAX/OTT OT coupling** (GPU-resident, no-copy — better than the host/POT couplers in
  torchcfm),
- **population-level held-out validation** (predict controls→target, score distribution metrics per
  condition vs an identity baseline),
- **binded** streaming/splitting.

**Generic (reuse the ideas / vendor the small math — do NOT hand-maintain, do NOT take heavy deps):**
- sinusoidal **time embeddings** (ott-jax / torchcfm formulas),
- **probability-path** math (flow-matching / OT-CFM literature),
- MLP/ResBlock backbones, FiLM conditioning.

The velocity-field *skeleton* (`v(t, x_t, cond)` MLP with time embedding) is itself generic; what makes it
*ours* is the condition (set) encoder + the surrounding pipeline. Treat the velocity **core** as
swappable/reusable and the **condition encoder + pipeline** as the contribution.

## 8. Lineage & attribution (credit, not dependency)

Several components are **torch ports of cellflow** (JAX/Flax), credited in-code:
- `SetEncoder` ← cellflow `ConditionEncoder` (`networks/_set_encoders.py`)
- `MLPVelocity` ← cellflow `ConditionalVelocityField` / `GENOTConditionalVelocityField`
  (`networks/_velocity_field.py`)
- objectives / metrics / data already cite cellflow (`OTFlowMatching`, `r_squared`, `compute_e_distance`)
- time features cite **ott-jax** and **torchcfm**; probability paths cite the flow-matching literature.

**This is attribution, not alignment.** We are *deliberately diverging* from cellflow toward a cleaner,
more maintainable torch design — we do not track its API and we never depend on it. (Note: some current
docstrings say "kept structurally aligned … mutually reviewable" — that wording overstates the intent and
should be softened to "adapted from … intentionally reparameterized.")

## 9. Reuse-vs-vendor policy (why we don't add diffusers / torchcfm / flow_matching)

A dependency earns its place only if it is large/complex, correctness-critical, actively maintained with a
*stable public* API, and shape-compatible. The reusable bits here (time embeddings, prob-path math, FiLM)
are tiny, settled math — **vendor them with attribution**, don't depend. Specifics, verified against the
live sources:
- **diffusers** ships image/latent models (`UNet2D…`, `DiT…`), semi-internal embedding modules, and a big
  transitive tail — too heavy for ~50 lines.
- **torchcfm** — its OT (`OTPlanSampler`) is POT/NumPy on the host, i.e. a *regression* vs our GPU JAX
  coupling. Keep ours.
- **Meta `flow_matching`** — clean path math (`AffineProbPath`, …) but young; API mismatch (`sample() ->
  PathSample`) needs an adapter.
- Our stack already has real lock/CUDA fragility (jax + torch + cupy + annbatch fork + binded on the
  cluster), so each new dep is unusually expensive here.

Exception that *did* earn its place: `huggingface-hub` + `safetensors` (real capability = sharing; light;
stable public API).

## 10. Swappable slots — `id (registry) | instance (in-memory)` (Component contract dropped)

**Decision (locked, 2026-07-22):** we did **not** build the registry/discriminator "Component contract"
(base + on-disk `{type, config}` discriminator + HF `coders`) originally sketched here — its only unique
win, a *generic* loader rebuilding a third-party class from a bare checkpoint, still needs that class
installed and isn't worth the "components can't be dataclasses" + on-disk-format tax. Instead a swappable
slot is typed `CombinerId | <BaseFamily instance> | None`, with **two** paths:

- **string id → per-family registry** (`combiner="concat"` / `"resnet1d"` / an extension's id). A small
  `{id → class}` registry per family (`COMBINER_REGISTRY` + `@register_combiner("id")`) replaces the old
  `if/elif` and is the **extension point**: an extension registers its own id from its own package (no PR
  to sc-flow). Serialization is **free** — the id string (+ a JSON `combiner_kwargs`) round-trips through
  `config.json`; on load the id is looked up and the class rebuilt. The **VF supplies the dims** (registered
  combiners take `(latent_state_dim, latent_time_dim, latent_condition_dim=None, **combiner_kwargs)`), so
  the user never sizes anything. An unregistered id raises a clear *"not registered — import the extension"*
  error (fine, and better than a silent default).
- **custom instance** (`combiner=MyCombiner(...)`) — the escape hatch for a throwaway class or one whose
  init needs **non-JSON** args. Usable in-memory (train/predict this session), but **`save_pretrained` is
  disabled** for it: `BaseModule.save_pretrained` raises if any slot holds a custom instance (detected via
  `_injected_submodule_slots()`), telling you to register it under an id to make it saveable. So instances
  never reach a checkpoint — no silent-drop, no re-supply dance.

The decision rule for an author: **JSON-serializable config → register it (round-trips, shareable); non-JSON
or throwaway → inject the instance (in-memory only, can't save).**

Belt-and-braces: `BaseModule._load_as_safetensor` still forces **`strict=True`** (the mixin default is
`False`, which would silently load a mismatched checkpoint) as a general safety net for any weight/arch
mismatch.

**Built and removed** (do not re-add without cause): the `Injectable` marker base, the
`{"__injected__": ...}` coder/marker + `save_pretrained` warning + `resolve_injected` ctor guard (the
"mark/warn/raise on load" variant), and the object-discriminator Component contract. **No discriminator, no
`{type, config}`, no class-path, no marker.** Reference impl: the `combiner` slot on `MLPVelocity`
(`CombinerId | BaseCombiner | None`) — verified: built-in id saves+loads; extension id round-trips via the
registry (id + kwargs in config, class rebuilt); custom instance works in-memory and `save_pretrained`
raises. Pooling and time-features stay on plain string-ids until given the same registry treatment.

## 11. cf-train migration (drop cellflow)

Target: `cf-train → sc-flow-tools → binded → annbatch(fork)`; **remove `cellflow-tools`**. Concretely:
1. Replace `cellflow-tools[annbatch]` with `sc-flow-tools` (pinned to an immutable commit / release).
2. Rewrite cf-train's model path off `cellflow.model.CellFlowAnnbatch` onto `sc_flow.FlowMatching` (or the
   `TrainingModule` + a registered `Objective`/`Predictor`).
3. Rename the stale `dagloader` imports → `binded`.
4. Regenerate `uv.lock` (fast/incremental) once the chain is clean. `uv` lives at `~/.local/bin/uv` on the
   cluster (not on `PATH`); `cf-train` sets a localscratch uv cache and seamless env.

## 12. Per-contributor playbook

- **New loss / training math** → implement an `Objective` in `flow` (or your extension), `@register_objective`.
  It may compute in torch or JAX (bridge with DLPack); the harness is unchanged.
- **New inference** (e.g. a different solver) → implement a `Predictor`, `@register_predictor`.
- **New architecture / backbone** → subclass `core.nn.BaseModule` → Hub-shareable for free. (Register only
  once the architecture registry is revived; today the VF is built directly.)
- **CFG / classifier-free guidance (pan-cellflow, Xiaotong)** → this was dropped in the port and is a known
  gap. Re-add as a first-class feature: condition-dropout + a null/`condition_null` embedding on the VF +
  the objective. Slots into the condition-encoder + objective seams.
- **Inverse-problem model (Lorenzo)** → a new `Objective` (+ `Predictor`), using the JAX bridge where you
  need ODE differentiation; share your pipeline config for review.
- **Fine-tuning popular models (Goncalo)** → the `core.nn.BaseModule` + Hub-sharing path is the entry;
  a foundation-models toolbox would sit as a sibling of `flow` on top of `mlcore`.
- **Running on HPC** → use `cf-train`; don't run on the login node.

## 13. Open decisions & roadmap

**In progress / next (Stage 3):**
- **Swappable slots — DONE (reference) / rolling out.** The Component contract was dropped in favour of
  `id (per-family registry) | instance (in-memory, save disabled) | None` (§10). Landed on the `combiner`
  slot of `MLPVelocity` (`COMBINER_REGISTRY` + `@register_combiner`; `save_pretrained` raises on a custom
  instance; strict-load net). Round-trips verified. *Next:* give
  **pooling** and **time-features** the same treatment when wanted. (Time-features rename to
  `sinusoidal`/`log-sinusoidal` — crediting ott-jax/torchcfm in comments only — is already done.)
- **`MLPVelocity` ctor** — still ~25 flat kwargs; the composable-sub-config cleanup is deferred (the slot
  polymorphism, which drove it, is now handled by injection, so this is lower priority).
- **Gaps to close:** attention pooling in `SetEncoder` (currently `NotImplementedError`); variable-length
  covariate-set masking; **CFG** (see §12).
- Wire `OptimConfig` into `configure_optimizers` + `fit` (optimizer is currently inline).
- Drop the dead `ARCHITECTURE_REGISTRY`; move the identity-baseline + debug logging out of
  `TrainingModule.validation_step` into a Lightning `Callback`; move the validation source-cap to the
  dataloader (there's a `TODO` in `validation_step`).
- **jax-optional packaging decision:** make `core` installable torch-only with jax/ott behind an extra
  (default `match_method="sinkhorn"` needs jax — decide default vs extra).
- Soften the cellflow "mutually reviewable" docstrings to "adapted-from, intentionally diverging" (§8).

**Publish path:**
1. Force-push `sc-flow-tools:feat/refactors` (human-run; classifier-gated).
2. Cut a `binded` release tag from `main` and repin downstream to the tag (over the raw commit).
3. Execute the cf-train migration (§11).
4. Extract `sc_flow.core → mlcore` (mechanical, once the seams settle).

## 14. Gotchas / footguns

- **HF mixin drops non-JSON `__init__` args silently** and loads weights **`strict=False`** by default, so
  a mismatch would load *silently* — a wrong model, no error. `BaseModule._load_as_safetensor` forces
  `strict=True` so this raises instead (§10). Tuples come back as **lists** on round-trip — normalize if you
  rely on tuple-ness.
- **Custom sub-module instances can't be saved** (§10). Slots take a registered string id *or* a custom
  instance; an instance isn't serializable, so `save_pretrained` **raises** (register it under an id to make
  it saveable). To carry hyperparameters through a registered id, they must be JSON (in `combiner_kwargs`).
- **Stringly-typed jax import**: `flow/_objectives.py` reaches the coupler via
  `require("sc_flow.flow.coupling._device")` — a string literal that refactors won't catch. Update it by
  hand on any move, and smoke-test (it only fires on the first OT step, deep into a run).
- **The test suite is largely stale** (many tests import modules the data-strip removed). The real gate is
  `scripts/smoke_train.py` + the active-import checks, not `pytest`.
- **Immutable pins** are a feature: never repin downstream to a mutable branch.

## 15. Pointers

- Model facade: `sc_flow/_model.py` (`FlowMatching`).
- Seams: `sc_flow/core/training/{_harness.py (TrainingModule), _objective.py, _predictor.py}`.
- FM impls: `sc_flow/flow/{_objectives.py, _predict.py (ODEPredictor), _vf.py, _set_encoder.py}`.
- Hub sharing: `sc_flow/core/nn/_modules.py` (`BaseModule`).
- Smoke test: `scripts/smoke_train.py` (`--objective otfm|genot`).
- Upstream lineage: cellflow `networks/`; ott-jax `time_encoder.py`; torchcfm `models/unet/nn.py`.
- binded: `theislab/binded` (`EvalLoader`, `Loader`, `Scheme`, `split_scheme`).

## 16. Glossary

- **Flow matching / CFM** — train a velocity field whose ODE transports a source distribution to a target;
  CFM = *conditional* flow matching (`x_t = (1-t)·x0 + t·x1`, regress the field onto `x1 - x0`).
- **VF (velocity field)** — the `nn.Module` `v(t, x_t, cond)` integrated (ODE) to translate cells.
- **OTFM** — OT flow matching: each minibatch, re-pair `(source, target)` by an optimal-transport plan
  before the CFM loss (cellflow's `OTFlowMatching`). `match_method="independent"` = vanilla CFM (no jax).
- **GENOT** — generative entropic OT: flow from latent **noise → target** with the (resampled) source cell
  *conditioning* the field (source is not the flow's start).
- **OT / OTT** — optimal transport; **OTT** = `ott-jax`, the JAX OT library used for the Sinkhorn coupling.
- **DLPack** — zero-copy tensor hand-off between torch and jax (no host round-trip) — how the torch batch
  reaches the JAX coupler and back.
- **CFG** — classifier-free guidance: train with random condition-dropout + a null/`condition_null`
  embedding so sampling can trade off conditional vs unconditional. **Dropped in the port; a known gap.**
- **FiLM** — feature-wise linear modulation, one of the conditioning mechanisms (`concatenation`/`film`/`resnet`).
- **SetEncoder / condition encoder** — permutation-invariant Deep-Sets encoder over the *set* of
  perturbation covariates → the condition embedding (mean, and log-variance if stochastic).
- **EvalLoader** — binded's control-rooted held-out eval reader: reads a control population **in full**,
  matched to perturbed targets by context (e.g. cell line).
- **binded / annbatch** — the streaming/splitting data layer / the out-of-core zarr sampler it's built on.
- **Objective / Predictor / TrainingModule** — the loss seam / the inference seam / the one generic
  Lightning harness that trains any `model` + `Objective` (+ optional `Predictor`).
