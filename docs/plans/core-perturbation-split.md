# Next-move plan: split the generic base out of the perturbation layer

**Status:** plan-only. Nothing here is implemented yet. Safe to start — no open maintainer
decisions block it. The one placement question (where the data layer lands) is already settled by the
pure-Python constraint in §0: top-level `sc_flow.data`.

**Audience:** the next agent/engineer picking this up cold. Read this whole file plus
[state.md](state.md) §3 (layered architecture) and §10 (component specs) before touching code.

**One-line goal:** make `sc_flow.core` a genuinely model-family-agnostic base (the future standalone
**`scfit`**) by evicting every perturbation / distribution-matching concept from it. The perturbation
data contract, the population metrics, and the source→target validation protocol move OUT of `core`
into the flow/perturbation layer. After this, extracting `scfit` is a mechanical `sc_flow.core →
scfit` lift, and `sc_flow` imports it.

---

## 0. Why (the decision behind this move)

`scfit` is meant to be a thin base that a **foundation-model** toolbox and the **flow-matching**
toolbox both sit on as siblings (state.md §3/§6). A foundation model doing masked-token pretraining
has **no source/target/coupling** — so it cannot and will not reuse a perturbation data contract.
Therefore anything that assumes source→target matching must not live in the shared base, or the FM
sibling inherits a data model it can't use.

Today `sc_flow.core` violates this: its `data/`, `metrics/`, and the `validation_step` of its
training harness are all perturbation-OT machinery wearing a "generic core" label. This move fixes
the layering inversion.

**Hard constraint discovered:** `import sc_flow.core.data` is contractually **pure-Python** — no
torch / jax / lightning ([_optional.py:3](../../src/sc_flow/_optional.py), enforced by the bare-env
import test). The relocated data layer must preserve that. Consequence: it lands as a **top-level
`sc_flow.data`**, a sibling of `core`/`flow`, NOT under `sc_flow/flow/` (whose package import pulls
torch). The population **metrics** are torch (`torchmetrics`) and perturbation-specific, so they go
under `flow/`, not the pure-Python data package.

---

## 1. Target structure (after this move)

```
sc_flow/
  core/     → the generic base (== future scfit). torch-only, no perturbation vocabulary.
              _component.py (ComponentRegistry)   — already generic, keep
              nn/            (MLP, Resnet1d, BaseModule, activation) — generic, keep
              training/      _objective.py, _predictor.py (seams)    — generic, keep
                             _harness.py            — KEEP but GENERICIZE (see step 4)
              _torch_types.py, _torch_utils.py      — generic, keep
  data/     → NEW top-level, pure-Python perturbation data contract (moved from core/data)
              FlowSpec, compile_obs, schemas/, containers/, _encoders, sim/, _mixins
  flow/     → OT / flow-matching toolbox (unchanged role) + two arrivals:
              metrics/       ← moved from core/metrics (RSquared, EnergyDistance, METRICS_REGISTRY)
              _validation.py ← NEW: the source→target / identity-baseline validation Callback
                               (extracted from core/training/_harness.py validation_step)
```

`sc_flow.core` after the move contains **zero** references to `source` / `target` / `condition` /
`coupling` / `control` / perturbation. That string check is the acceptance test for the whole move.

---

## 2. Classification — what is generic vs perturbation (evidence)

**Generic → stays in `core` (future scfit):**
- `core/_component.py` — `ComponentSpec`/`ComponentRegistry`, family-agnostic.
- `core/nn/` — `_modules.py` (MLP/Resnet1d/BaseModule), `_activation.py`, `_utils.py`. Backbones.
- `core/training/_objective.py`, `_predictor.py` — abstract seams + registries, no family assumptions.
- `core/training/_harness.py` — the loop is generic; **only `validation_step`/`on_validation_epoch_end`
  leak** (see step 4).
- `core/_torch_types.py`, `core/_torch_utils.py` — generic torch helpers.

**Perturbation / distribution-matching → moves to `sc_flow.data` (pure-Python):**
- `core/data/_spec.py` — `FlowSpec` ("data spec for a flow-matching problem").
- `core/data/_compile_obs.py` — source/control vs target/perturbed nodes, coupling, `condition_fn`.
- `core/data/schemas/` — `_condition_`, `_coupling_`, `_covariates_`, `_state_`, `_base_schema` (kept
  ones only, see step 1 pruning).
- `core/data/containers/_categorical.py` — the surviving container.
- `core/data/_encoders.py` — categorical/lookup/one-hot encoders (single-cell `.uns`-tied; only
  consumer is the perturbation schemas — moves with them).
- `core/data/sim/_dummy_adata.py` — perturbation fixture (drugs/KOs/controls).
- `core/data/_mixins.py` — `BatchMixin`/`MappedArray` (audit; keep only what containers use).

**Perturbation metrics → moves to `sc_flow.flow.metrics` (torch):**
- `core/metrics/_metrics.py` — `RSquared` (per-condition perturbation R²), `EnergyDistance`
  (scPerturb E-distance), `METRICS_REGISTRY`.

**Dead — DELETE, do not move (from the design-review pass):**
- `core/data/_abc.py` — `Distribution` / `MatchedDistributions` ABCs: zero implementations, zero
  callers; only `DataTree`/`DataT` are used (by `_mixins.py`). Also a TypeVar-name bug at
  `_abc.py:19` (`TypeVar("DistributionDType")` bound to alias `DistributionT`). Keep only `DataTree`/
  `DataT` if `_mixins` still needs them; drop the rest.
- `core/data/_utils.py` — `get_covariates_encoders_from_dict` / `get_covariate_encoder` /
  `get_label_encoder` / `get_one_hot_encoder` / `get_functional_encoder`: duplicate the sklearn
  wrappers in `_encoders.py` and are unreferenced. Keep only `convert_to_categorical_in_place`.
- `core/data/schemas/_base_schema.py` — `get_data`/`extract_array`/`_extract_array`/`_verify_schema`/
  `_get_data` are never called (`compile_obs` reads columns itself). Reduce the schema ABC to the
  property bag that is actually consumed.
- `core/data/schemas/_response_data_schema.py` — `ResponseDataSchema` is exported but never used by
  `compile_obs`/`FlowSpec`. Orphan; delete.
- `core/data/containers/_base.py` — `BaseData` ABC: audit; `CategoricalData` is the sole subclass, so
  most methods (`slice_with_index`, `concat_collection`, …) may be inlineable/removable.

Pruning **first** (step 1) means we move less code and the relocation diff is honest.

---

## 3. Call sites that must update (grep-verified)

Eager import to re-home in the package root:
- [`sc_flow/__init__.py:13`](../../src/sc_flow/__init__.py) — `from sc_flow.core import data` → `from
  sc_flow import data` (and update the module docstring lines 6–7 about `core.data`).
- [`_optional.py:3`](../../src/sc_flow/_optional.py) — docstring says `import sc_flow.core.data` is
  pure-Python; update to `sc_flow.data`. The guarantee itself must still hold.

Model facade (belongs to the flow layer — these become `sc_flow.data` / `sc_flow.flow` imports):
- [`_model.py:20`](../../src/sc_flow/_model.py) `from sc_flow.core.data import FlowSpec`,
  `:23` `CompiledDims`/`DataInput`, `:397` `from sc_flow.core.metrics import METRICS_REGISTRY`,
  and doc refs at `:93`, `:200`, `:507`.

Scripts (mechanical path rewrite `sc_flow.core.data`→`sc_flow.data`, `sc_flow.core.metrics`→
`sc_flow.flow.metrics`):
- `scripts/smoke_train.py:30-32`, `scripts/tahoe_eval.py:72-75`, `scripts/profile_train.py:54-56`,
  `scripts/tahoe_train.py:65-67`, `scripts/tahoe_smoke.py:39-41`.

Internal `data/` self-references (all the `sc_flow.core.data.*` imports inside the moved files —
schemas, containers, compile_obs, encoders) rewrite to `sc_flow.data.*`. `logger` name at
`_compile_obs.py:47` too.

---

## 4. Genericize the harness (the one non-mechanical step)

[`core/training/_harness.py:105-152`](../../src/sc_flow/core/training/_harness.py)
(`validation_step` + `on_validation_epoch_end`) hardcodes perturbation eval: it reads
`batch["source"]/["target"]/["leaf"]`, subsamples a control population, and scores an
identity-baseline (`_id_metrics`, logged as `<name>_identity`). A generic harness must not know a
batch has source/target.

Move this protocol into a **flow-layer Lightning `Callback`** (`sc_flow/flow/_validation.py`). The
generic harness keeps only: "given a `Predictor` and `val_metrics`, run `predict` and `update` the
metrics against a caller-supplied target" — expressed without naming `source`/`target`/`control`.
The identity-baseline + source-cap + `debug_val` logic is perturbation-specific and lives in the
Callback. This also resolves the existing `# TODO` at `_harness.py:108` (the source cap "is a
dataloader concern, not the training module's") — the cap moves out of `core` entirely.

Wiring change: `_model.py` (the flow facade) constructs the harness with the Callback instead of
passing `val_metrics`/`predictor`/`val_max_source_cells`/`debug_val` into a perturbation-aware
`TrainingModule`. Decide the minimal generic validation surface the harness still exposes (probably:
`predictor` + `val_metrics` scored against `batch[<target-key>]` where the target key is supplied by
the Callback, or the Callback owns the whole `on_validation_batch_end`).

---

## 5. Order of operations

1. **Prune dead strata** in `core/data` (§2 "Dead") — smallest safe diff first; re-run smoke gate.
2. **Genericize the harness** (step 4): extract the flow validation Callback, strip perturbation from
   `core/training/_harness.py`, rewire `_model.py`. Gate on `smoke_train.py` (both objectives) —
   validation metrics must match before/after.
3. **Move `core/data` → `sc_flow/data`** (top-level, pure-Python). Update internal + external imports
   (§3). Re-run the **bare-env pure-Python import test** — `import sc_flow.data` must not pull torch.
4. **Move `core/metrics` → `sc_flow/flow/metrics`.** Update `_model.py` + `tahoe_eval.py`.
5. **String-check acceptance:** `grep -rn "source\|target\|condition\|coupling\|control\|perturb"
   src/sc_flow/core` returns nothing meaningful (only generic uses, if any).
6. **Docs:** update state.md §3/§5 layout + §13 roadmap to reflect `sc_flow.data` as a pure-Python
   sibling and the flow validation Callback.

---

## 6. Verification gates

- `scripts/smoke_train.py --objective otfm` and `--objective genot` — trains + predicts + learns the
  shift; validation metrics unchanged vs. pre-move.
- `scripts/smoke_component_specs.py` — registry / spec round-trip still green.
- Bare-env import test — `import sc_flow.data` (and `import sc_flow`) with torch/jax **absent** must
  succeed (the `_optional.py` contract).
- Final: `import sc_flow.core` must not transitively import `sc_flow.data`, `sc_flow.flow`, or
  `torchmetrics`-backed metrics — the base stands alone.

---

## 7. Out of scope (separate tasks, note the ordering)

- **Objective/Predictor → `ComponentRegistry` migration** (state.md §10; the "Q1" plan): converts the
  old `dict`+`build_*(name,*args,**kwargs)` registries to typed config + build-context. Independent of
  this move; do it before wiring entry-point plugins.
- **Entry-point plugin discovery** (`mlcore.components`): depends on the registry migration; makes the
  OT/flow layer a discovered plugin rather than a direct import.
- **Portable data serialization** (`input_schema.json` + `assets.safetensors` replacing the
  cloudpickled `condition_fn`): the data layer's persistence is still closure/pickle-shaped; that
  rewrite happens after this move, inside the new `sc_flow.data`.
- **`sc_flow.core → scfit` extraction**: only after `core` is provably standalone (§6 final gate).
  Then `scfit` is a new distribution and `sc-flow-tools` depends on it; drop the in-repo copy.
</content>
</invoke>
