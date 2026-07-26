"""PROOF: every family (flow_matching, foundation) is driven by ONE application-level path.

The whole point of the symmetric-plugin design is that the app special-cases no paradigm. This script *is*
the app's runner in miniature: it imports **only** ``sc_flow.families`` (never a model module), DISCOVERS a
family by name, and drives it with a single generic ``app_run`` — the exact shape of cf-train's ``myapp.train._run``
(base ``Trainer`` kwargs + the family's ``trainer_overrides``; read the family's ``metrics_history`` after fit).

If ``flow_matching`` needed a bespoke branch, this file could not train it with the same function that trains
``foundation``. It can. Run: ``python scripts/proof_family_symmetry.py`` (sc-flow-tools venv, CPU-only).
"""

import sys

# The "app" imports the registry ONLY — no sc_flow._model, no sc_flow.concept, no torch yet.
from sc_flow.families import available_families, build_family

assert "torch" not in sys.modules and "sc_flow._model" not in sys.modules, "app pulled in a model at import"
FAMILIES = available_families()
print("available_families():", FAMILIES, "(torch-free)")
assert {"flow_matching", "foundation"} <= set(FAMILIES), FAMILIES
assert "pancell" not in FAMILIES, "pancell is a composition, not a family"

import anndata as ad  # noqa: E402
import lightning.pytorch as pl  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import sparse  # noqa: E402


def app_run(family: str, recipe: dict, *, max_steps: int) -> dict:
    """The ONE generic runner — identical for every family. Mirrors myapp.train._run exactly."""
    builder = build_family(family, recipe)  # discovery loads the plugin HERE; the app imported no model
    for attr in ("module", "datamodule", "callbacks", "save", "metrics_history", "trainer_overrides"):
        assert hasattr(builder, attr), f"{family}: builder is missing contract member {attr!r}"
    kwargs = dict(
        max_steps=max_steps, accelerator="cpu", devices=1, callbacks=list(builder.callbacks),
        logger=False, enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
    )
    kwargs.update(getattr(builder, "trainer_overrides", {}) or {})  # the family's own eval cadence
    print(f"  [{family}] trainer_overrides={getattr(builder, 'trainer_overrides', {})}")
    pl.Trainer(**kwargs).fit(builder.module, datamodule=builder.datamodule)
    return dict(getattr(builder, "metrics_history", {}) or {})


# --- family A: flow_matching (held-out perturbation eval → non-empty metrics_history) ------------------
rng = np.random.default_rng(0)
n = 1500
pert = pd.DataFrame({"cell_line": rng.choice(list("ABC"), n), "drug": rng.choice(["d0", "d1", "d2", "d3"], n),
                     "is_control": False})
ctrl = pd.DataFrame({"cell_line": rng.choice(list("ABC"), 400), "drug": "control", "is_control": True})
obs = pd.concat([pert, ctrl], ignore_index=True)
obs["cell_line"] = obs["cell_line"].astype("category")
obs["drug"] = obs["drug"].astype("category")
flow_adata = ad.AnnData(X=rng.standard_normal((len(obs), 16), dtype=np.float32), obs=obs)
flow_adata.obs_names = [str(i) for i in range(len(obs))]
flow_recipe = {
    "data": {"data": flow_adata, "sample_rep": "X", "control_key": "is_control",
             "perturbation_covariates": {"drug": ["drug"]}, "split_covariates": ["cell_line"],
             "split_by": ["drug"], "split_ratios": {"train": 0.75, "val": 0.25}},
    "model": {"objective": "otfm", "match_method": "independent",
              "pooling": {"type": "sc_flow.mean", "version": 1, "config": {}},
              "hidden_dims": [8], "condition_embedding_dim": 8, "state_latent_dim": 8,
              "time_latent_dim": 4, "num_time_features": 16},
    "sampler": {"batch_size": 32, "chunk_size": 1, "prefetch_factor": 2},
    "trainer": {"num_iterations": 6, "valid_freq": 3, "device": "cpu",
                "metrics": ["mean_aggregated_r_squared", "e-dist"], "val_num_steps": 3,
                "val_max_source_cells": 128, "lr": 1e-4},
}
print("\n== flow_matching (via build_family + generic app_run) ==")
flow_hist = app_run("flow_matching", flow_recipe, max_steps=6)
print("  metrics_history keys:", sorted(flow_hist))
assert flow_hist and any(v for v in flow_hist.values()), "flow validation did not populate metrics_history"

# --- family B: foundation (contrastive pretrain; no eval yet → empty metrics_history) -----------------
N_CELLS, N_GENES, N_PROG = 256, 120, 12
programs = [rng.choice(N_GENES, 20, replace=False) for _ in range(N_PROG)]
cell_prog = rng.integers(0, N_PROG, N_CELLS)
x = np.zeros((N_CELLS, N_GENES), dtype=np.float32)
for i in range(N_CELLS):
    rate = np.full(N_GENES, 0.2)
    rate[programs[cell_prog[i]]] += rng.uniform(2.0, 5.0)
    x[i] = rng.poisson(rate * rng.uniform(0.5, 1.5)).astype(np.float32)
found_adata = ad.AnnData(X=sparse.csr_matrix(x),
                         obs=pd.DataFrame({"cell_type": pd.Categorical([f"program_{p}" for p in cell_prog])}))
found_adata.var_names = [f"ENSG{i:011d}" for i in range(N_GENES)]
found_adata.obs_names = [str(i) for i in range(N_CELLS)]
found_recipe = {
    "data": {"adata": found_adata, "species": "hsapiens"},
    "backbone": {"dim_model": 64, "n_layers": 2, "n_heads": 4, "dim_feedforward": 128, "dropout": 0.0, "max_rank": 65},
    "objective": {"logit_scale_init": 3.0, "max_logit_scale": 100.0},
    "sampler": {"batch_size": 64, "max_tokens": 64}, "trainer": {"lr": 1e-3}, "task": "contrastive", "seed": 0,
}
print("\n== foundation (via build_family + the SAME generic app_run) ==")
found_hist = app_run("foundation", found_recipe, max_steps=5)
print("  metrics_history keys:", sorted(found_hist))
assert found_hist == {}, f"foundation has no eval yet; expected empty history, got {found_hist}"

print("\nFAMILY SYMMETRY PROOF PASSED — one app path drove flow_matching AND foundation via discovery.")
