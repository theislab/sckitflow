# Roadmap — the multi-repo vision

**Status:** plan-only. Master index for the planning docs in this folder. Read this first, then the
detail docs it links.

## Vision in one paragraph

One **data layer** (`binded`) turns an AnnData + a declarative spec into a training-batch stream, and
feeds several **model libraries** that each keep their own framework and training loop:
**sc-flow-tools** (torch flow-matching), **cellflow** (jax flow-matching), and a **scConcept** foundation
model. Each model is trained through a shared **training template** (`ml-template`: a
framework-agnostic entrypoint + pydantic/YAML orchestration + submitit/Optuna) that carries the
model's own **`RunConfig`** (omegaconf-structured) inside. Model libraries never depend on each other;
they share only `binded` (data) and the template (orchestration). Cross-scale link between a
foundation model and a flow model is a single `obsm` key.

## Layered architecture

```
 orchestration (training template) ── ml-template harness: pydantic+YAML, submitit, Optuna, Snakemake
        │   framework-agnostic entrypoint `python -m app.train`  → loads outer config, calls model.from_config
        ▼
 model libraries (each owns its loop) ── sc-flow-tools (torch) · cellflow (jax) · scConcept (torch/Lightning FM)
        │   each carries an omegaconf `RunConfig`; consumes a batch stream; owns methods/loops internally
        ▼
 data layer ─────────────────────────── binded:  ProblemSpec protocol  +  providers
        FlowProblemSpec (schema → dagloader.Scheme, annbatch)   {source,target,condition}
        ConceptProblemSpec (wraps scConcept tokenizer/collate)  {view_1,view_2}
```

## Repos (all under `/Users/selman/projects/`)

| Repo | Role | Framework | This roadmap's change |
|---|---|---|---|
| **binded** (`theislab/binded`, fresh 2026-07-16) | **data layer** for all model libs — evolved `dagloader` + `ProblemSpec` | numpy/annbatch (+ optional torch/lamin) | **NEW** — populate from `dagloader`; see [binded-spec.md](binded-spec.md) |
| **dagloader** (authoritative: `cellflow/src/cellflow/data/dagloader` @ `feat/annbatch-loader`) | current streaming core (3 diverged copies — dedup target) | numpy/annbatch | seeds `binded`; standalone `/projects/dagloader` is stale, ignore |
| **sc-flow-tools** (this repo) | torch flow-matching library | torch | consume `binded`; torch-only collapse; see [data-layer-separation.md](data-layer-separation.md) |
| **cellflow** (`/projects/cellflow`) | jax flow-matching library | jax/optax | consume `binded` (drop vendored `dagloader`) |
| **scConcept** (`/projects/scConcept`) | single-cell foundation model | torch/Lightning | data via `binded`; standardized training app; see [scconcept-rewrite.md](scconcept-rewrite.md) |
| **ml-template** (`/projects/ml-template`) | **the training template** (harness) | agnostic | the reusable orchestration pattern the training apps instantiate |
| **cf-train** / **train-cellflow** | cellflow's orchestration app (an ml-template instance) | agnostic | build `ProblemSpec` from config; run any model |

Full ecosystem detail + key files per repo: [data-layer-separation.md](data-layer-separation.md) §1.

## Config idiom — the two layers (this was the source of confusion)

There are **two** config layers; they compose, they are not competitors:

- **Outer / orchestration config** — `ml-template` style: **pydantic v2 + layered YAML**, owns run
  provenance, paths, submitit resources/profiles, sweeps. Lives in the *training app* (cf-train, a
  future scConcept-train). Unknown keys = loud error.
- **Inner / model config** — **`RunConfig`, omegaconf-structured dataclasses**, defined by the *model
  library* (sc-flow-tools already has this; scConcept will get one). Describes data spec + method +
  optimizer + trainer for that model.

They meet at the framework-agnostic entrypoint: the harness loads the outer YAML, hands the `model`
block to `Model.from_config(RunConfig)`. So orchestration and model evolve independently.

## Workstreams (ordered by dependency)

**WS1 — `binded` (data layer).** Foundational; blocks WS2–WS5. Absorb `dagloader`'s streaming core;
add the `ProblemSpec` protocol + `Batch`/`batch_keys`; add `FlowProblemSpec` (sc-flow schemas →
`dagloader.Scheme`); stub `ConceptProblemSpec`. Detail: [binded-spec.md](binded-spec.md).

**WS2 — sc-flow-tools refactor.** Consume `binded` for data (retire `DataManager.compile_data`
indexing); keep methods/loops internal; begin the **torch-only collapse** (make `backends/jax`
retirement a tracked follow-on, drop the `backend=` axis from config/resolve). Detail:
[data-layer-separation.md](data-layer-separation.md) + [sc-flow-refactor.md](sc-flow-refactor.md).

**WS3 — cellflow refactor.** Replace the vendored `src/dagloader` with a dependency on `binded`;
confirm it consumes `FlowProblemSpec` output unchanged (near-free — it already reads
`{source,target,condition}`). Keep the jax/optax loop. Detail:
[sc-flow-refactor.md](sc-flow-refactor.md) §cellflow.

**WS4 — scConcept parallel training app.** A new ml-template instance that imports scConcept's model +
tokenizer as a *library*, wires data through `binded.ConceptProblemSpec`, configures via an omegaconf
`RunConfig`, runs under the harness with a Lightning loop. **Rewrite the harness, not the science.**
Detail: [scconcept-rewrite.md](scconcept-rewrite.md).

**WS5 — orchestration convergence.** Point cf-train (and a new scConcept-train) at
`ProblemSpec`-from-config so one entrypoint can run any model. Keep cf-train "just runs stuff".

### Dependency graph

```
WS1 binded ──┬──► WS2 sc-flow-tools ──┐
             ├──► WS3 cellflow ────────┼──► WS5 orchestration
             └──► WS4 scConcept-app ───┘
```

Do **WS1 first**. WS2/WS3/WS4 can proceed in parallel once `binded` exposes `ProblemSpec` +
`FlowProblemSpec`. WS4 additionally needs `ConceptProblemSpec`.

## Cross-cutting locked decisions (recap)

Contract is **data-only**; methods/models/**loops** stay internal per library (loops genuinely
diverge). sc-flow-tools goes **torch-only** (cellflow is the jax path). "FM" = **foundation models**
(a 2nd task family; Lightning/FSDP earns its place there). Cross-scale = pretrained scConcept
`obsm['X_scConcept']` consumed as a flow `sample_rep`. Full list + rationale:
[data-layer-separation.md](data-layer-separation.md) §2.

## Deferred (not in this roadmap yet)

`Method.fit()` / training-loop abstraction; the **ODE inverse-problem** loop (A: train-through-ODE
method vs B: per-instance posterior over a frozen field — fork unresolved); torchax cross-framework
gradients; the 80 stale-test-double failures.
