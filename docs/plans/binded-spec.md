# `binded` — the data-layer repo (spec)

**Status:** plan-only; repo is fresh/greenfield (`theislab/binded`, created 2026-07-16). Part of the
[roadmap](roadmap.md). This is **WS1** and blocks the rest.

## What it is

`binded` is the **evolved `dagloader`**: the streaming core stays, plus a small `ProblemSpec` contract
and per-family providers, so multiple model libraries share one data layer. Name comes from `Bind`
(source↔target binding). It is the *only* thing sc-flow-tools, cellflow, and the scConcept training
app share on the data side.

- **Seed code (authoritative):** `cellflow/src/cellflow/data/dagloader` on branch
  **`feat/annbatch-loader`** — NOT the standalone `/projects/dagloader` (stale, ignore). This copy has
  **diverged**: it adds `_eval_loader.py` + `_split.py` and drops `_scheduled_sampler.py`/`_schemes.py`,
  so its public surface differs from the old standalone — read it directly at WS1 time before assuming
  the API below. Note there are **three** dagloader copies today (standalone,
  `cellflow/src/dagloader`, `cellflow/src/cellflow/data/dagloader`) — consolidating them into `binded`
  is itself part of the point (the group's "don't duplicate, define an interface" thesis).
- **Also seeds from:** `sc-flow-tools/src/sc_flow/data/schemas/*` (the "label part" → `FlowProblemSpec`).
- **scConcept** dataloading (`lamin_dataloader` tokenizer, two token-views) → `ConceptProblemSpec`
  adapter (optional extra; do **not** reimplement the tokenizer).

## Scope

**In:** the streaming core (`Scheme`/`Node`/`Bind`/`Weights`/`SamplerConfig`/`DAGLoader`/
`ScheduledClassSampler`/schemes); the `ProblemSpec` protocol + `Batch`/`batch_keys`; `FlowProblemSpec`
(schema → `Scheme`); a provider seam + `ConceptProblemSpec` adapter (optional).

**Out:** models, methods, training loops, `RunConfig` (those belong to the model libs); anything
framework-specific beyond thin optional provider adapters.

## Proposed package layout

```
binded/
  src/binded/
    __init__.py                 # re-export core + contract
    _schema.py  _loader.py       # ← from dagloader (Scheme/Node/Bind/Weights/SamplerConfig/DAGLoader)
    _scheduled_sampler.py _schemes.py _io.py
    contract/
      _spec.py                   # ProblemSpec protocol, Batch, batch_keys
    providers/
      flow.py                    # FlowProblemSpec  (schemas → Scheme → DAGLoader)  [extra: flow]
      flow_schemas/              # ← moved from sc-flow-tools data/schemas (State/Condition/Coupling/Groups)
      concept.py                 # ConceptProblemSpec (wraps scConcept datamodule/collate) [extra: concept]
  README.md                      # keep dagloader's "Batch contract" + "Mental model" sections
```

## Public API

**Core (from dagloader, unchanged):** `DAGLoader`, `Scheme`, `Node`, `Bind`, `Weights`,
`SamplerConfig`, `ScheduledClassSampler`, `perturbation_scheme`, `uniform`/`frequency`/
`inverse_frequency`.

**Contract:**
```python
class ProblemSpec(Protocol):
    def validate(self, adata) -> None: ...
    def build_loader(self, adata, sampler_cfg) -> Iterable[Batch]: ...
    @property
    def batch_keys(self) -> frozenset[str]: ...
    def to_dict(self) -> dict: ...
    @classmethod
    def from_dict(cls, d: dict) -> "ProblemSpec": ...
```

**Providers:**
- `FlowProblemSpec(sample_rep=..., conditions=..., split_covariates=...)` → `{source,target,condition}`
  (matched) or `{target}` (single-node, no `Bind`). Built on the moved sc-flow schemas.
- `ConceptProblemSpec(...)` → `{view_1, view_2}` each `{tokens, values}`. Thin wrapper over
  scConcept's `AnnDataModule`/`Collate`/`MultiSpeciesTokenizer`.

## Dependencies & extras

Keep the core light; gate heavy deps behind extras so each consumer pulls only what it needs:
- `binded` (core) → numpy, annbatch.
- `binded[flow]` → adds whatever the schemas need (anndata).
- `binded[concept]` → adds `lamin_dataloader` + torch + the scConcept adapter (or takes scConcept as
  an optional dep).

Dependency direction (target): `sc-flow-tools`, `cellflow`, `scConcept-train` each depend on
`binded`. **No model lib depends on another.**

## Migration from `dagloader`

Authoritative source = `cellflow/src/cellflow/data/dagloader` @ `feat/annbatch-loader`. The standalone
`/projects/dagloader` and `cellflow/src/dagloader` are stale — do **not** seed from them.

1. Port the authoritative copy's modules into `binded/src/binded/` (package → `binded`); keep its tests.
2. **cellflow:** replace the in-tree `src/cellflow/data/dagloader` (and its imports) with a dependency
   on `binded`. Delete the second copy `src/cellflow/src/dagloader`.
3. **cf-train:** update `from dagloader import SamplerConfig` → `from binded import SamplerConfig`.
4. Decide compatibility: a `dagloader` shim re-exporting `binded` for one release, or a hard cut
   (small blast radius — only cellflow + cf-train import it today).

## What moves out of sc-flow-tools into binded

The **label half** of the data layer: `data/schemas/*` (`StateDataSchema`, `ConditionDataSchema`,
`CouplingDataSchema`, `GroupsDataSchema`, `ResponseDataSchema`) → `binded/providers/flow_schemas/`,
updated to emit a `dagloader.Scheme`. The **`compile_data` half** of `sc_flow/data/_manager.py`
(`_get_mapped_index`→`MappedLevelIndex`, `_get_matched_distributions`) is **retired**, not moved.

## Next steps to populate

1. Scaffold `binded` from the cookiecutter/`ml-template` conventions the group uses (pyproject,
   ruff, tests, CI).
2. Port `dagloader` core verbatim; keep its tests green.
3. Add `contract/_spec.py` (`ProblemSpec`, `Batch`, `batch_keys`) with unit tests.
4. Port the sc-flow schemas → `FlowProblemSpec` with the `schema → Scheme` translation; integration
   test (schema → one batch of the right shape).
5. Stub `ConceptProblemSpec` (interface only) so WS4 can fill it.
6. Switch cellflow + cf-train imports; delete the vendored copy.
