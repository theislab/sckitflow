# legacy/ — archival, not installed

This directory holds subsystems that were quarantined off the active training path and then moved out of
the installed package (`src/sc_flow/`) entirely. It is kept in the repository for reference only:

- **Not installed** — the wheel ships only `src/sc_flow`, so nothing here is importable from an installed
  `sc_flow`.
- **Not on any import path** — the active toolbox (`sc_flow.core`, `sc_flow.flow`, `FlowMatching`) does not
  import anything here.
- **Not runnable as-is** — these modules still reference pre-quarantine module paths (`sc_flow.methods`,
  `sc_flow.trainer`, `sc_flow.config`, `sc_flow.backends`, …) that no longer exist, so they will not import
  without rewiring. They are historical artifacts, not maintained code.

Contents include the old JAX subsystems (`jax_*`), old torch solvers/methods/coupling/surrogate, and the
earlier `config` / `trainer` / `dataset` / `preprocessing` / `external` layers. The stack is deliberately
moving off these (torch-native weights, no cellflow/JAX for weights); see `docs/plans/state.md`.

`git log --follow` on any file here recovers its full history from when it lived under `src/sc_flow/legacy/`.
