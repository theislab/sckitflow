# Next-move plan: separate the data layer into a shared `ProblemSpec` contract

**Status:** plan-only. Nothing here is implemented yet. Do not write code until the "Open
decisions" below are answered by the maintainer (@selmanozleyen).

**Audience:** the next agent/engineer picking this up cold. This doc carries all cross-repo context
so you do **not** need to re-derive it. Read the "Ecosystem" and "Locked decisions" sections first.

**Part of the [roadmap](roadmap.md).** The contract package is now named **`binded`**
(`theislab/binded`) — the evolved `dagloader`. This doc is the *data-contract detail*; see
[binded-spec.md](binded-spec.md) for the repo, [sc-flow-refactor.md](sc-flow-refactor.md) for the
sc-flow/cellflow changes, and [scconcept-rewrite.md](scconcept-rewrite.md) for the FM training app.

**One-line goal:** extract the *data* concern (AnnData → training batch stream) into a small,
provider-agnostic `ProblemSpec` contract shared by multiple model libraries — with **dagloader**
(flow/perturbation) and **scConcept** (tokenized contrastive) as the first two providers. Everything
else (methods, models, training loops) stays *internal* to each library.

---

## 1. Ecosystem — the repos and what each is

All under `/Users/selman/projects/`. Frameworks and roles matter; do not conflate them.

| Repo | Path | Framework | Role |
|---|---|---|---|
| **sc-flow-tools** (this repo) | `sc-flow-tools` | torch (+ a jax backend, candidate to retire) | The **torch flow-matching** library. Will become *one implementer* of the data contract, not "the framework". Branch: `feat/refactors`. |
| **cellflow** | `cellflow` | **JAX** / flax / optax | A *complete, standalone* flow-matching model library (OT-FM, GENOT). A **peer** that will *consume* the data contract. Keeps its own jax loop. Vendors dagloader under `src/dagloader`. |
| **dagloader** | `dagloader` | plain numpy/annbatch | Declarative, index-free streaming sampler over [annbatch](https://github.com/laminlabs/annbatch). **Half of the data contract already exists here.** |
| **scConcept** | `scConcept` | torch + Lightning | Single-cell **foundation model** (contrastive pretraining, HF model hub). Cloned 2026-07-16. Its **dataloading** is the second provider we must support. Uses `lamin_dataloader`, NOT dagloader. |
| **cf-train** | `cf-train` | framework-agnostic | **Orchestration / config-home only.** Its `train.py` imports `cellflow.model.CellFlowAnnbatch` and just runs it. "Its job is only to run stuff." Not a model library. |
| **train-cellflow** | `train-cellflow` | — | Older/sibling variant of cf-train (`src/train_cellflow/run`). Same pattern. |

### Key files to read (with why)

**sc-flow-tools** — the data classes that become the flow provider:
- `src/sc_flow/data/_manager.py` — `DataManager`. **Split this**: keep the schema/label half; retire
  the `compile_data` half (`_get_mapped_index` → `MappedLevelIndex`, `_get_matched_distributions`,
  `_get_distribution_data`) — dagloader replaces in-memory index materialization with streaming.
- `src/sc_flow/data/schemas/` — `_base_schema.py` (`DataSchema` ABC), `_state_data_schema.py`
  (`StateDataSchema`; `sample_rep`), `_condition_data_schema.py` (`ConditionDataSchema`; `conditions`,
  `conditions_reps`, `conditions_covariates`), `_coupling_data_schema.py`, `_groups_data_schema.py`,
  `_response_data_schema.py`. **These are the "label part" to keep and update to emit a
  `dagloader.Scheme`.**
- `src/sc_flow/data/containers/`, `.../samplers/`, `.../grouping/` — supporting types; audit for what
  survives once streaming replaces indexing.
- `src/sc_flow/trainer/_trainer.py` — `Trainer.train()` (~L106-156). Shows the current **hard 1:1
  loop** (`train_step` → `opt_manager.step`, single optimizer). Context for why loops are NOT shared.
- `src/sc_flow/backends/torch/methods/library/_cfm.py` — a concrete flow method; shows the composable
  axes (coupling, path, velocity field, `step_fn`=loss, solver). Not touched by this move, but read it
  to understand what *consumes* the batch.
- `src/sc_flow/config/_capabilities.py` (`MethodCapabilities`), `src/sc_flow/config/_resolve.py`
  (`{"kind": ...}` → object registries; note the `backend` param that collapses if we go torch-only).

**dagloader** — the batch contract (already written):
- `src/dagloader/__init__.py` — public API: `DAGLoader`, `Scheme`, `Node`, `Bind`, `Weights`,
  `SamplerConfig`, `ScheduledClassSampler`, `perturbation_scheme`, `uniform/frequency/inverse_frequency`.
- `src/dagloader/README.md` — **read the "Batch contract" and "Mental model" sections.** Batch =
  `{"target": (B,d), "source": (B,d), "condition": (B,e)}`, one condition per batch, source↔target
  matched by context via `Bind`. `Scheme` = structure (columns+weights+binds); `SamplerConfig` = read
  params (batch/chunk/preload). Index-free.
- `src/dagloader/_schema.py`, `_loader.py`, `_scheduled_sampler.py`, `_schemes.py`.

**scConcept** — the second dataloading provider:
- `src/concept/data/datamodules.py` — `AnnDataModule(lightning.LightningDataModule)`. Substrate is
  `lamin_dataloader` (`TokenizedDataset`, `Tokenizer`, `LaminDiskCollection`) + `torch.utils.data`
  `DataLoader`/`StatefulDataLoader`. **Not annbatch, not dagloader.**
- `src/concept/data/collate.py` — `Collate.__call__(batch)` emits **two augmented views**, each item
  `{"tokens", "values"}` (variable-length gene tokens, panel dropout, QC mask). Contrastive.
- `src/concept/data/samplers.py` — `WithinGroupSampler`, `DistributedSamplerWrapper`.
- `src/concept/dataset` — `MultiSpeciesTokenizedDataset`, `MultiSpeciesTokenizer`.
- `src/concept/conf/` — Hydra: `config.yaml` composes `datamodule/*` + `model/ContrastiveModel.yaml`;
  per-species/per-corpus split files under `datamodule/split_*`.
- README use cases: `load_config_and_model` (HF or local), `extract_embeddings(adata) →
  result['cls_cell_emb']` written to `adata.obsm['X_scConcept']`, `concept.train(adata, ...)` (light
  adapt), large-scale pretrain (Hydra + distributed). Python **3.12+**, optional flash-attn.

**cf-train** — orchestration reference:
- `src/myapp/train.py` (`run(cfg) -> float`; imports `cellflow.model.CellFlowAnnbatch`,
  `dagloader.SamplerConfig`), `config.py` (pydantic `Config`), `data.py` (`build_source`),
  `run/{launch,sweep,executor}.py`. `configs/base.yaml` + `configs/experiments/*` + `configs/profiles/*`.

---

## 2. Locked decisions (from the design conversation, 2026-07-16)

1. **Option A — contract + peers, not a merge.** sc-flow-tools defines/owns as little as possible;
   cellflow and scConcept stay independent libraries that *comply with* / *feed* the contract. No
   absorbing cellflow (jax) or scConcept (tokenizer/FM) into this repo.
2. **The contract is scoped to DATA only** (this move). Method / Model / training-loop / capabilities
   /registry protocols are **NOT** promoted to a cross-library contract — they stay internal to each
   library. Rule: *something becomes contract only when a second library actually needs to
   interoperate on it.* Data has ≥3 consumers (sc-flow, cellflow, cf-train) and now a 2nd batch shape
   (scConcept) → it qualifies. Methods/loops do not.
3. **Training loops are per-method, not shared.** Proven: cellflow = jax/optax loop; sc-flow simple
   methods = torch 1:1 loop; the ODE **inverse problem** = its own loop (simulate-in-step, adjoint
   gradients, possibly optimize inputs not weights). A `Method` owns `fit()`; `step_fn` is a
   convenience of the *simple* family only. (Deferred — not part of this move.)
4. **sc-flow-tools becomes torch-only.** cellflow is the jax implementer. The internal
   `backends/torch` vs `backends/jax` split and the `backend=` axis threaded through configs/resolve
   can then collapse — this is the root cause of the "too many ids in the interfaces" problem.
   (Retirement of `backends/jax` is a *follow-on*, not required by this move; note it.)
5. **"FM" = Foundation Models.** Two task *families* over the shared data substrate: flow-matching and
   foundation-model. FM training fits the `general` method category (custom `train_step`), not the
   flow skeleton. Lightning (FSDP) genuinely earns its place for the FM family — unlike wrapping
   cellflow's jax.
6. **Cross-scale bridge = one obsm key.** A pretrained scConcept produces
   `adata.obsm['X_scConcept']`; a flow `ProblemSpec` consumes it as `sample_rep`. No code coupling
   between the libraries — just a representation key.
7. **ProblemSpec is a PROTOCOL, not a field schema.** A single field-based spec cannot express both
   `{source,target,condition}` (dense, matched) and `{view_1,view_2}` (tokenized, augmented). The
   stable seam is the *boundary* (`AnnData → batch stream` + declared `batch_keys`), with one concrete
   spec per family.

### Already done earlier on `feat/refactors` (do not redo)
- pyproject/env cleanup: extras `test` / `test-torch` / `test-jax` / `test-all` (backend deps
  de-duplicated via self-referencing extras); missing `flax`/`pot`/`torchdiffeq`/`torchsde` fixed.
- Python floor raised to **3.11**. CI split into core (hatch matrix, both backends) + torch job
  (`tests/backends/torch`) + jax job (`tests/backends/jax`), each installing only its deps.
- Fixed one cross-backend test leak (`tests/backends/jax/solvers/test_sde_solvers.py` imported a torch
  type).
- Known-open: ~80 pre-existing core-test failures from **stale test doubles** drifting from current
  abstract interfaces (`extract_state_data` now abstract; `PredictionData.X`). Tracked separately;
  unrelated to this move.

---

## 3. The contract to build

### 3.1 Batch contracts (the shapes)

| Family | `batch_keys` | Shape | Provider |
|---|---|---|---|
| Flow (matched) | `{source, target, condition}` | dense `(B,d)/(B,d)/(B,e)` | dagloader |
| Flow (unmatched) | `{target}` (single stream) | dense `(B,d)` | dagloader single-node, no `Bind` |
| Concept (contrastive) | `{view_1, view_2}` each `{tokens, values}` | variable-length token seqs | scConcept / lamin_dataloader |

### 3.2 The `ProblemSpec` protocol (the stable seam)

```python
class ProblemSpec(Protocol):
    def validate(self, adata) -> None: ...                 # required obs/var/obsm keys exist
    def build_loader(self, adata, sampler_cfg) -> Iterable[Batch]: ...   # AnnData → this family's stream
    @property
    def batch_keys(self) -> frozenset[str]: ...            # what a batch contains
    def to_dict(self) -> dict: ...                         # serialisable (reproducibility / config home)
    @classmethod
    def from_dict(cls, d: dict) -> "ProblemSpec": ...
```

Consumers (methods) declare which `batch_keys` they need; a generic check
`method.consumes <= spec.batch_keys` validates spec↔consumer compatibility *without the framework
knowing what a velocity field or a token is*.

### 3.3 The two providers

- **`FlowProblemSpec`** — built from sc-flow-tools' existing schemas.
  `schemas (conditions / conditions_covariates / split_covariates / sample_rep)` → `dagloader.Scheme`
  + `SamplerConfig` → `DAGLoader`. Mapping: schema condition/split columns → `Node.cols`; source↔target
  context → `Bind.common`; selection → `Weights`. Retire `DataManager.compile_data` indexing.
- **`ConceptProblemSpec`** — **wrap, do not reimplement.** Adapt scConcept's `AnnDataModule` /
  `Collate` / `MultiSpeciesTokenizer` behind the same protocol. Do NOT rebuild the tokenizer, panel
  sampling, or multi-species vocab. Likely lives with/near scConcept or takes it as an optional dep.

---

## 4. Where things live (dependency direction)

Target: neither model library depends on the other; both take a **light** dependency on the data
contract.

```
             ┌────────────────────────── data contract (small, pure-ish) ──────────────────────────┐
             │  ProblemSpec protocol + Batch types + batch_keys                                     │
             │  provider: FlowProblemSpec  (schemas → dagloader.Scheme)      → depends on dagloader  │
             │  provider: ConceptProblemSpec (wraps scConcept dataloading)   → optional/soft dep     │
             └───────────────────────────────────────────────────────────────────────────────────┘
                    ▲                         ▲                          ▲
       sc-flow-tools (torch flow)     cellflow (jax flow)          cf-train (orchestration)
       consumes Flow batches          consumes Flow batches         builds spec from config, runs
```

**RESOLVED (2026-07-16):** the contract + `FlowProblemSpec` live in **`binded`** (`theislab/binded`,
option (a) — the evolved `dagloader`). `sc-flow-tools`, `cellflow`, and the scConcept training app all
depend on `binded`; no model lib depends on another. Full repo spec: [binded-spec.md](binded-spec.md).

---

## 5. Execution steps (once decisions are answered)

1. **Confirm the batch contract** in code: a typed `Batch` (TypedDict/dataclass) + `batch_keys`, plus
   the `ProblemSpec` protocol. Land in the chosen home package.
2. **`FlowProblemSpec`:** move sc-flow schemas over; add `schema → dagloader.Scheme` translation;
   `build_loader` returns a `DAGLoader`. Cover matched *and* single-node unmatched.
3. **Retire `compile_data`:** delete/relocate `_get_mapped_index` / `_get_matched_distributions`
   indexing from `DataManager`; make `DataManager` (if kept) a thin holder of a `ProblemSpec` + a
   loader, or drop it in favor of `FlowProblemSpec` directly.
4. **Point sc-flow-tools' methods/trainer at the new loader** (batches now come from `DAGLoader`, not
   the old sampler/index path). Keep method/loop code internal & unchanged in shape.
5. **`ConceptProblemSpec`:** wrap scConcept's `AnnDataModule`/`Collate` behind the protocol; emit
   `{view_1, view_2}`. Keep scConcept as the source of the tokenizer.
6. **cellflow:** verify it consumes `FlowProblemSpec.build_loader(...)` output unchanged (it already
   consumes dagloader-style `{source,target,condition}`; this should be near-free).
7. **cf-train:** have `data.py`/`config.py` construct a `ProblemSpec` from config and hand its loader
   to whichever model — replacing bespoke `build_source` wiring.
8. **Tests:** contract-level tests (a `ProblemSpec` builds a valid stream; `batch_keys` correct;
   round-trip `to_dict/from_dict`); a flow integration test (schema → dagloader → one batch of the
   right shape); a concept smoke test (wrapped scConcept emits two views).

---

## 6. Explicitly OUT of scope for this move (deferred)

- Method / Model / capabilities / registry as a **cross-library** contract (stays internal).
- The training-loop abstraction / `Method.fit()` refactor; the **ODE inverse-problem** loop
  (simulate-in-step + adjoint). Note: the repo's `SchrodingerBridgeProbabilityPath` is the
  *simulation-free bridge-matching* variant (analytic `compute_ut`) and does NOT need a special loop —
  it's just a path swap. The loop-different case is the *through-ODE / per-instance inverse* method.
  Still-open sub-decision recorded for later: is the inverse problem (A) a train-through-ODE **method**
  (owns a custom plan) or (B) a per-instance posterior procedure over a **frozen** field (an
  inference-time op, like `extract_embeddings`)? This fork decides where it lives.
- Retiring `backends/jax` from sc-flow-tools (follow-on once cellflow is confirmed as the jax path).
- torchax / cross-framework gradient bridge (only needed for a torch loss over a jax net — the chosen
  design avoids it).
- The 80 stale-test-double failures (separate cleanup).

---

## 7. How to get oriented fast (commands)

```bash
# batch contract (read this first)
sed -n '1,60p' /Users/selman/projects/dagloader/src/dagloader/README.md
# the schemas to keep/convert
ls /Users/selman/projects/sc-flow-tools/src/sc_flow/data/schemas/
# the compile_data half to retire
grep -n "compile_data\|_get_mapped_index\|_get_matched_distributions" \
  /Users/selman/projects/sc-flow-tools/src/sc_flow/data/_manager.py
# scConcept dataloading substrate (the 2nd provider)
sed -n '1,60p' /Users/selman/projects/scConcept/src/concept/data/datamodules.py
```
