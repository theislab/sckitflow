# Reference artifacts — streaming data path design

Concrete, standalone artifacts backing [`../streaming_data_path.md`](../streaming_data_path.md).
These are **design references**, not part of the `sc_flow` package.

| file | what it is |
|---|---|
| `verify_scflow_to_scheme.py` | Runnable verification: compiles an `sc_flow`-vocabulary spec → a **real** `dagloader.Scheme`, drives the **real** `DAGLoader`, asserts the streamed cells obey the compiled structure (target/source split + `Bind` matching + `condition_fn`). 16/16 pass in a venv with `dagloader` + `annbatch` + anndata ≥0.13. `condition_fn` is a one-hot stand-in for the real schemas (see doc §10). |
| `compiler_sketch.py` | The `lower_to_scheme` compiler as 8 annotated lowering passes (`DataManager` config → `Scheme`/`Node`/`Bind` + `condition_fn`), plus the `StepData` model-ABI adapter. Uses `store` (not `source`) for containers. |
| `scflow_data_core_extracted.py` | The `sc_flow.data` core with the container layer stripped, showing the `Distribution`/`MatchedDistributions`/`DataTree` seam and where `dagloader` matches (multi-input + container-agnostic). |

To run the verification: use an environment where `dagloader` and `annbatch` are importable
(currently cellflow's venv), then `python verify_scflow_to_scheme.py`.
