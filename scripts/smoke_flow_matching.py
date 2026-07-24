"""Local CPU smoke for the Lightning-native path: recipe -> FlowMatching builder -> Trainer.fit.

No sc_flow.data, no fit() facade. FlowMatching builds scfit Streams+Loader + the LightningModule +
datamodule + validation callback; training is plain pl.Trainer.fit on a synthetic in-memory AnnData.

Run: ``python scripts/smoke_flow_matching.py`` (sc-flow-tools venv, CPU-only).
"""

import anndata as ad
import lightning.pytorch as pl
import numpy as np
import pandas as pd

from sc_flow import FlowMatching

rng = np.random.default_rng(0)
cell_lines, drugs = ["A", "B", "C"], ["d0", "d1", "d2", "d3"]
pert = pd.DataFrame({
    "cell_line": rng.choice(cell_lines, 1500),
    "drug": rng.choice(drugs, 1500),
    "is_control": False,
})
ctrl = pd.DataFrame({"cell_line": rng.choice(cell_lines, 400), "drug": "control", "is_control": True})
obs = pd.concat([pert, ctrl], ignore_index=True)
obs["cell_line"] = obs["cell_line"].astype("category")
obs["drug"] = obs["drug"].astype("category")
adata = ad.AnnData(X=rng.standard_normal((len(obs), 16), dtype=np.float32), obs=obs)
adata.obs_names = [str(i) for i in range(len(obs))]

recipe = {
    "data": {
        "data": adata,                       # in-memory source (paths on the cluster)
        "sample_rep": "X",
        "control_key": "is_control",
        "perturbation_covariates": {"drug": ["drug"]},
        "split_covariates": ["cell_line"],   # match context
        "split_by": ["drug"],                # hold out whole drugs
        "split_ratios": {"train": 0.75, "val": 0.25},
    },
    "model": {
        "objective": "otfm",
        "match_method": "independent",       # no JAX OT solve on CPU
        "pooling": {"type": "sc_flow.mean", "version": 1, "config": {}},
        "hidden_dims": [8],
        "condition_embedding_dim": 8,
        "state_latent_dim": 8,
        "time_latent_dim": 4,
        "num_time_features": 16,
    },
    "sampler": {"batch_size": 32, "chunk_size": 1, "prefetch_factor": 2},
    "trainer": {"num_iterations": 6, "valid_freq": 3, "device": "cpu",
                "metrics": ["mean_aggregated_r_squared", "e-dist"], "val_num_steps": 3,
                "val_max_source_cells": 128, "lr": 1e-4},
}

fm = FlowMatching(recipe)
print(f"[smoke] built: state_dim={fm._dims.state} drug_vocab={fm._dims.condition_num_categories} "
      f"callbacks={len(fm.callbacks)}")

trainer = pl.Trainer(
    max_steps=6, accelerator="cpu", devices=1, val_check_interval=3, num_sanity_val_steps=0,
    limit_val_batches=4, callbacks=fm.callbacks, logger=False, enable_checkpointing=False,
    enable_progress_bar=False, enable_model_summary=False,
)
trainer.fit(fm.module, datamodule=fm.datamodule)

hist = fm.callbacks[0].metrics_history if fm.callbacks else {}
print("[smoke] metrics_history:", {k: [round(x, 3) for x in v] for k, v in hist.items()})
assert hist and any(v for v in hist.values()), "validation must populate metrics_history"
print("\nNEW-PATH SMOKE PASSED")
