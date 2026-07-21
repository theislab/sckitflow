# Training template + scConcept parallel rewrite (spec)

**Status:** plan-only. Part of the [roadmap](roadmap.md), **WS4**. Needs `binded` (WS1) —
specifically `ConceptProblemSpec`.

## The "training template" — what it is

`/Users/selman/projects/ml-template` **is** the training template. It is a *framework-agnostic
training application* pattern:

- **framework-agnostic entrypoint** `python -m app.train` (the keystone) — every orchestrator calls
  it; the training loop stays whatever the model needs (torch/Lightning, jax, scverse).
- **typed config:** pydantic v2 + layered YAML (`configs/{base,experiments,profiles}`), unknown keys
  = loud error, config = provenance.
- **submit/resources/requeue+resume:** submitit (`run/`), laptop↔cluster by swapping `--profile`.
- **sweeps:** Optuna over the same entrypoint; **DAG (optional):** Snakemake calls the same entrypoint.
- **caching:** content-hashed paths.

`cf-train` and `train-cellflow` are existing **instances** of this template wrapping cellflow.

## The two config layers (how they compose)

Not competitors — they nest:

| | Outer (orchestration) | Inner (model) |
|---|---|---|
| Tool | **pydantic v2 + YAML** (ml-template) | **`RunConfig`, omegaconf** (model lib) |
| Owns | run provenance, paths, submitit profiles, sweeps | data spec + method + optimizer + trainer |
| Lives in | the training *app* (cf-train, scConcept-train) | the model *library* (sc-flow-tools; scConcept gets one) |

The entrypoint loads the outer YAML and hands its `model` block to `Model.from_config(RunConfig)`.
Orchestration and model then evolve independently. This is the pattern to standardize across all
model libs.

## Why a scConcept parallel rewrite

Today scConcept ships its own bespoke Hydra config tree (`src/concept/conf/*`), custom datamodule
wiring, and its own CLI. The parallel version standardizes it onto the shared stack **without
touching the science**:

| Concern | scConcept today | Parallel version |
|---|---|---|
| data | `lamin_dataloader` + custom `AnnDataModule`/`Collate` | `binded.ConceptProblemSpec` (wraps the same tokenizer/collate) |
| config | bespoke Hydra tree | omegaconf `RunConfig` (inner) + ml-template YAML (outer) |
| orchestration | custom scripts | ml-template harness (submitit/Optuna/Snakemake) |
| loop | Lightning | **keep Lightning** (FSDP/precision/checkpoint — earns its place for an FM) |
| model / tokenizer / flash-attn | — | **unchanged**, imported as a library |

**Principle: rewrite the harness, not the model.** Mirror the cellflow/cf-train split — scConcept
stays the model library; a new `scConcept-train` app (an ml-template instance) provides config + data
+ orchestration.

## Shape of the new `scConcept-train` app

```
scConcept-train/                     # a fresh ml-template instance
  configs/{base,experiments,profiles}/…   # outer: pydantic+YAML (species, corpus, resources)
  src/app/
    train.py     # framework-agnostic entrypoint → builds RunConfig → concept model + Lightning Trainer
    config.py    # outer pydantic Config
    data.py      # constructs binded.ConceptProblemSpec, hands loader to the Lightning module
    run/{launch,sweep}.py
  # imports:  concept (model, tokenizer, modules) as a library ;  binded[concept] for data
```

`ConceptProblemSpec` (in `binded`, WS1 stub → filled here) wraps scConcept's
`MultiSpeciesTokenizer`/`Collate`/`AnnDataModule` and emits `{view_1, view_2}`. Note scConcept's
substrate is `lamin_dataloader`, **not** annbatch/dagloader — so this provider is an *adapter*, gated
behind `binded[concept]`.

## Open decisions

1. **New repo `scConcept-train`, or a `train/` app inside scConcept?** Recommend a **separate repo**
   (keeps the model library import-light and publishable to HF, like scConcept is today; matches
   cellflow/cf-train separation).
2. **How much of scConcept's model API to touch:** ideally none — add only a thin
   `from_config(RunConfig)` constructor if its current entry isn't config-first. Keep tokenizer,
   contrastive modules, flash-attn untouched.
3. **Python floor:** scConcept requires **3.12+**; the shared stack targets **3.11**. The scConcept-train
   app can pin 3.12 independently (it's a separate app), but note the mismatch for any shared code in
   `binded[concept]`.

## Next steps

1. Land `binded.ConceptProblemSpec` (interface in WS1, implementation here).
2. Scaffold `scConcept-train` from `ml-template`.
3. Define scConcept's inner `RunConfig` (data spec + model + optim + trainer); add
   `from_config` if needed.
4. Wire `data.py` → `ConceptProblemSpec` → Lightning module; smoke-run light adaptation on a small
   AnnData (mirrors `concept.train(adata, ...)`).
5. Add submitit profiles + one Optuna sweep over the entrypoint.
