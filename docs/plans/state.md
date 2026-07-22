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

**Hub sharing (foundation-model story).** The current code mixes
`huggingface_hub.PyTorchModelHubMixin` into `core.nn.BaseModule`, but automatic `__init__` capture is **not**
the serialization contract we will standardize: it silently drops non-JSON arguments, and strict
`state_dict` loading cannot detect parameter-free semantic changes (for example, an activation changing
from `Tanh` to `ReLU`). The target is one explicit, versioned config on the **top-level exportable model**;
leaf modules remain ordinary `nn.Module`s. The top-level model may still use the Hub mixin for
`save_pretrained` / `from_pretrained` / `push_to_hub`, but passes its config explicitly and stores weights
as safetensors. We deliberately do **not** adopt the HF `Trainer` (its supervised eval loop fights our
population-level generative validation); only the artifact transport is reused. See §10.

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

## 10. Serialization + extensibility — explicit component specs (decision accepted)

**Decision (accepted, 2026-07-22; implementation incremental):** portable models use an explicit,
versioned **component discriminator**. A runtime object and its persistent description are different
things: the object is an `nn.Module`/`Objective`/`Predictor`; the description is immutable, JSON-safe data.
We will not rely on HF constructor auto-capture, pickle a Python object graph, or store arbitrary importable
class paths in a portable model.

The deliberately small wire contract is:

```python
class ComponentSpec(TypedDict):
    type: str                  # stable, namespaced id, e.g. "sc_flow.concat"
    version: int              # config schema version for this component
    config: dict[str, JsonValue]
```

Terminology: **config** is any JSON configuration data, and is unfortunately also the conventional name
for a whole model's configuration file. A **spec** is specifically a self-identifying component config:
the `{type, version, config}` envelope. The inner `config` contains only parameters understood by that
`type` at that `version`; it cannot select an implementation by itself. In Python, `ComponentSpec` /
`PoolingSpec` are `TypedDict`s because they describe that JSON mapping directly. The factory may parse the
inner mapping into a private frozen dataclass for validation and convenient typed access, but that parsed
object is not an additional artifact representation.

The **slot determines the family** (`combiner`, `architecture`, `objective`, `predictor`, `encoder`, …), so
the wire object does not repeat it. Each stable family uses the same generic registry implementation:
`type → {config_type, build, migrate, provider}`. A registered config type may be a frozen dataclass; the
runtime component does **not** need to be a dataclass or inherit a serialization base. The factory receives
a `BuildContext` containing derived dimensions/device information, so a user still never manually sizes a
combiner or condition encoder.

Example nested model config (illustrative names):

```json
{
  "format_version": 1,
  "architecture": {
    "type": "sc_flow.mlp_velocity",
    "version": 1,
    "config": {
      "activation": "tanh",
      "combiner": {"type": "sc_flow.concat", "version": 1, "config": {}},
      "condition_encoder": {
        "type": "sc_flow.set_encoder",
        "version": 1,
        "config": {"output_dim": 64, "pooling": "mean"}
      }
    }
  }
}
```

This is the same basic pattern used successfully elsewhere:

- **[Hugging Face Transformers](https://huggingface.co/docs/transformers/model_doc/auto):**
  `config.json.model_type` selects a registered `Config`/`Model` pair via `AutoConfig`/`AutoModel`. We
  adapt the stable discriminator + typed-config + factory pattern.
- **[Keras](https://keras.io/guides/serialization_and_saving/):** nested objects serialize as a registered
  name plus JSON config and are rebuilt through a custom-object registry. We adapt the recursive
  registered-object pattern, but not Python module/class paths.
- **[Diffusers](https://huggingface.co/docs/diffusers/api/configuration):** models/components carry JSON
  configs and pipelines have a component manifest. We adapt the bundle/manifest separation, not its
  dependency or constructor-capture implementation.
- **Hydra/OmegaConf:** remains useful for trusted experiment composition and CLI overrides. A Hydra
  `_target_`/Python class path is **not** the portable checkpoint format.

We adapt these **contracts**, not their code: this should be a small local implementation, not a dependency
on Transformers, Keras, or Diffusers.

### Extension path

- Built-ins register directly in their owning package.
- A third-party package exposes registrations through the standard
  [Python package-entry-point](https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/)
  group `mlcore.components`; loading discovers installed providers without requiring a magical
  `import my_extension` first.
- IDs are namespaced and stable (`my_package.my_component`), never bare globally-contended names such as
  `resnet`.
- The manifest records the providing distribution/version. An unavailable component raises an error that
  names the missing provider; an old config version is migrated explicitly or rejected.
- Not every choice becomes an **external** plugin. Parameter-free architecture choices (activation, time
  features) must be serialized but may remain validated enums. A polymorphic trainable submodule gets a
  discriminated spec even when its registry is initially closed to built-ins. The public entry-point
  budget is initially: top-level architecture, objective, predictor, and fitted data encoder/transform;
  nested families are exposed to third-party entry points only when cross-package extension is genuinely
  required.

### Attention pooling: specified component, not necessarily public plugin

Attention pooling is the motivating nested example. `mean`/`sum` are parameter-free, while token- and
seed-attention pooling add weights and have different output-dimension and hyperparameter contracts. They
therefore cannot be represented safely as `pooling="..."` plus an untyped kwargs bag. `SetEncoderConfig`
contains a built-in discriminated `PoolingSpec` from the first implementation slice:

```json
{"type": "sc_flow.attention_token", "version": 1,
 "config": {"num_heads": 8, "qkv_dim": 64, "dropout": 0.0, "num_layers": 1,
            "transformer_block": false, "layer_norm": false, "ff_dim": null}}
```

Built-in variants are `sc_flow.mean`, `sc_flow.sum`, `sc_flow.attention_token`, and
`sc_flow.attention_seed`. This is a closed built-in family initially; it can use the same registry
machinery without promising third-party pooling compatibility forever.

**Implemented decision (July 2026):** `PoolingSpec` is a `TypedDict` describing the JSON object itself—not a
second dataclass wrapper around the wire representation. Its factory performs runtime validation, parses the
component-specific config into internal typed dataclasses, and builds the torch module from the closed
built-in registry. There are **no hidden architecture defaults**: `type`, `version`, and `config` are
required, attention configs must contain every
v1 field (including explicit `false`, `0.0`, and `null` values), and no string alias or omitted pooling choice
is accepted. `BasePooling.forward(x, mask=None) -> Tensor` is the runtime interface over
`x: (batch, set, features)` and an explicit boolean `mask: (batch, set)`; every implementation exposes
`input_dim` and `output_dim`. `SetEncoder(pooling=PoolingSpec)` is portable, while
`SetEncoder(pooling=BasePooling(...))` is the experimental polymorphic path. The latter trains normally but
`save_pretrained` fails before writing because subclassing establishes tensor compatibility, **not** a
reconstruction contract. Promotion to a portable component later means assigning a stable namespaced type,
JSON config schema/version, factory registration, and tests; it does not require changing checkpoint JSON.

The export guard is centralized in `BaseModule`, not opt-in per injectable subclass. `BaseModule.__new__`
records every `torch.nn.Module` instance supplied as a constructor argument (also inside mappings and
sequences), and `save_pretrained` reports the qualified argument paths before writing. Subclasses must not
implement `_injected_submodule_slots`-style reporting hooks: forgetting an override would make the safety
property fail open. This guard is a transitional backstop, not the serialization design itself. In the
completed design, the portable construction path accepts a spec and constructs its runtime module inside
the factory; passing an already-built module unambiguously selects the runtime-only research path.

Lineage/reuse decision: `cellflow` owns working JAX `TokenAttentionPooling` and
`SeedAttentionPooling` implementations. `CellFlow2` on `dedup/reexport-cellflow` already imports/re-exports
those classes instead of maintaining copies, which is the right same-backend deduplication pattern.
`sc-flow-tools` remains torch-only for model weights and must not depend on cellflow, so it ports the
algorithms and configuration semantics with attribution rather than importing the JAX modules. Prefer
torch's maintained attention primitives over transliterating cellflow's manual head splitting.

Masking is part of the pooling interface, not inferred serialization trivia. The torch port accepts an
explicit per-example valid-token mask from the compiled biological input schema/data path; it must not
infer padding solely from `embedding == 0`, and must not assume every example shares the first example's
mask. Required tests: permutation invariance, mixed-length masks in one batch, padded-value independence,
defined all-masked behaviour, output-dimension validation, eval/dropout determinism, and fresh-process
config + weights round-trip.

### Custom Python escape hatch and the two artifact tiers

A custom instance is still useful for research, but it must not blur two different persistence promises:

1. **Trusted resume checkpoint** — Lightning/PyTorch checkpoint + optimizer/RNG state + exact experiment
   config + git commit/lockfile. Custom Python instances/class paths are allowed because the original code
   environment is required. Expensive HPC work must always be checkpointable.
2. **Portable pretrained bundle** — full component specs + safetensors + biological input schema/assets +
   manifest. No pickle, closure, arbitrary callable, or out-of-band `from_pretrained(..., component=...)`
   argument. Export fails early and reports every unresolved runtime override.

Thus a throwaway component may train and resume without registration; it becomes publishable/shareable
only after receiving a stable id and JSON config. `strict=True` weight loading remains mandatory, but is a
last safety net, **not** proof that the architecture is the same: parameter-free operations and reordered
biological features can change semantics without changing state-dict keys.

### Portable bundle and biological schema

The target artifact is:

```text
config.json             versioned mathematical model/component graph
model.safetensors       parameters and persistent tensor buffers
input_schema.json       ordered features + modality/identifier/condition semantics
assets.safetensors      fitted lookup/PCA/scaler arrays (with keys described by the schema)
manifest.json           format, package/plugin versions, hashes, provenance
README.md               model card / usage
```

`input_schema.json` must record at least the ordered feature identifiers (not just `state_dim`), identifier
namespace/organism where relevant, expected AnnData representation, fitted transforms, condition columns
and vocabularies, lookup keys, and control/null-condition semantics. The current cloudpickled
`condition_fn` is replaced by a deterministic compiler from this saved schema + assets.

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
- **New architecture / backbone** → implement an ordinary torch `nn.Module`, a JSON-safe typed config,
  and an architecture factory registered under a stable namespaced `ComponentSpec.type` (§10). Do not add
  HF serialization to the leaf module.
- **CFG / classifier-free guidance (pan-cellflow, Xiaotong)** → this was dropped in the port and is a known
  gap. Re-add as a first-class feature: condition-dropout + a null/`condition_null` embedding on the VF +
  the objective. Slots into the condition-encoder + objective seams.
- **Inverse-problem model (Lorenzo)** → a new `Objective` (+ `Predictor`), using the JAX bridge where you
  need ODE differentiation; share your pipeline config for review.
- **Fine-tuning popular models (Goncalo)** → wrap the upstream model as a top-level registered
  architecture + explicit config and export the common portable bundle; a foundation-models toolbox would
  sit as a sibling of `flow` on top of `mlcore`.
- **Running on HPC** → use `cf-train`; don't run on the login node.

## 13. Open decisions & roadmap

**In progress / next (Stage 3):**
- **Component specs + portable bundle — accepted, not built.** Replace the experimental
  `CombinerId | instance` serialization path with the explicit discriminator/factory contract in §10.
  First slice: generic `ComponentRegistry`, `ComponentSpec`, `BuildContext`, and a fully round-tripping
  `MLPVelocityConfig` containing `SetEncoderConfig` + built-in `PoolingSpec` + `CombinerSpec`. Then replace
  `FlowMatching.save/load` (`state.pkl`) with the portable bundle and add entry-point discovery.
- **`MLPVelocity` ctor — stream embedders DONE; full bundle pending.** The three MLP stream maps are now
  presence-based `MLPEmbedderConfig | None` slots (`state_embedder` / `time_embedder` / `source_embedder`;
  `None` = raw stream, a config = MLP), replacing the inconsistent `encode_*` bool vs kwargs-presence
  toggles. Renamed `*_encoder → *_embedder` (DiT `x_embedder`/`t_embedder` precedent; `proj` was rejected —
  it connotes a single Linear, these are multi-layer — and it disambiguates the Deep-Sets
  `condition_encoder`). `MLPEmbedderConfig` is a plain dataclass that round-trips via the HF mixin. Still to
  do: fold these + `SetEncoderConfig` / `PoolingSpec` / `CombinerSpec` into the single `MLPVelocityConfig`
  bundle so training and loading share one factory (per the component-spec slice above).
- **Attention pooling DONE:** explicit per-realm, per-example masks now flow through `SetEncoder`,
  `MLPVelocity`, objectives, and prediction; the attributed torch token/seed ports plus mean/sum use the
  closed `PoolingSpec` registry. Today's compiled condition schemas have fixed, dense slot counts, so an
  omitted mask explicitly means "dense/unpadded" and takes a genuinely mask-free fast path: no all-ones
  tensor, mask concatenation, or attention-mask operation is performed. If any realm is masked, masks for
  every pooled realm are required. If/when the compiler accepts genuinely ragged biological sets, it must
  emit `batch["condition_mask"]` rather than infer validity from zero embeddings. **CFG** remains open
  (see §12).
- Wire `OptimConfig` into `configure_optimizers` + `fit` (optimizer is currently inline).
- Drop the current dead, training-only `ARCHITECTURE_REGISTRY`; the new generic component registry owns
  artifact reconstruction. Move the identity-baseline + debug logging out of
  `TrainingModule.validation_step` into a Lightning `Callback`; move the validation source-cap to the
  dataloader (there's a `TODO` in `validation_step`).
- **jax-optional packaging decision:** make `core` installable torch-only with jax/ott behind an extra
  (default `match_method="sinkhorn"` needs jax — decide default vs extra).
- Soften the cellflow "mutually reviewable" docstrings to "adapted-from, intentionally diverging" (§8).

**Publish path:**
1. Land the component-spec contract + portable round-trip tests before presenting serialization as stable.
2. Force-push `sc-flow-tools:feat/refactors` (human-run; classifier-gated).
3. Cut a `binded` release tag from `main` and repin downstream to the tag (over the raw commit).
4. Execute the cf-train migration (§11).
5. Extract `sc_flow.core → mlcore` only after the artifact and extension seams settle.

## 14. Gotchas / footguns

- **HF constructor capture is not architecture serialization.** It silently drops non-JSON arguments;
  defaults may then be written in their place. Even `strict=True` cannot catch a parameter-free semantic
  change: a saved `MLP(activation_cls=Tanh)` can rebuild with `ReLU`, match every state-dict key, and return
  different predictions. Only the explicit config graph in §10 closes this class of bug.
- **Do not confuse resume with portable export.** A runtime override/custom instance may always be included
  in a trusted experiment checkpoint with its code environment. Portable export must fail before writing
  if any component, callable, fitted transform, or biological input semantic lacks a `ComponentSpec` or
  schema entry.
- **`FlowMatching.save/load` currently uses cloudpickle.** Treat existing `state.pkl` files as trusted-code
  artifacts only; never load one from an untrusted source. Migrate them to the §10 bundle rather than
  extending the pickle format.
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
- Current experimental Hub sharing: `sc_flow/core/nn/_modules.py` (`BaseModule`); target contract: §10.
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
