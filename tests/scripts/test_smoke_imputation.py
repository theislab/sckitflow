# tests/test_smoke_fit_cfm_panel.py
import os
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def tiny_panel_adata(tmp_path_factory):
    """Synthetic adata with the obsm keys + obs columns the panel config needs."""
    rng = np.random.default_rng(0)
    out = tmp_path_factory.mktemp("adata") / "adata_concat.h5ad"
    
    n_old, n_new = 200, 200
    n_obs = n_old + n_new
    
    # shared markers, then panel-specific
    shared = [f"shared_{i}" for i in range(10)]
    old_only = [f"old_{i}" for i in range(5)]
    new_only = [f"new_{i}" for i in range(5)]
    
    var_names = shared + old_only + new_only
    n_vars = len(var_names)
    
    X = rng.normal(size=(n_obs, n_vars)).astype(np.float32)
    # zero-out panel-exclusive markers in the opposite panel
    for j, vn in enumerate(var_names):
        if vn in old_only:
            X[n_old:, j] = 0.0
        elif vn in new_only:
            X[:n_old, j] = 0.0
    
    obs = pd.DataFrame(
        {
            "dataset": ["old panel"] * n_old + ["new panel"] * n_new,
            "split": (["train"] * int(n_old * 0.8) + ["test"] * (n_old - int(n_old * 0.8))
                     + ["train"] * int(n_new * 0.8) + ["test"] * (n_new - int(n_new * 0.8))),
        },
        index=[f"cell_{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame(index=var_names)
    
    adata = ad.AnnData(X=X, obs=obs, var=var)
    
    # obsm keys the panel config reads
    adata.obsm["X_zscore"] = rng.normal(size=(n_obs, 12)).astype(np.float32)
    adata.obsm["X_lin"] = rng.normal(size=(n_obs, 12)).astype(np.float32)
    adata.obsm["X_quad_src"] = rng.normal(size=(n_obs, 6)).astype(np.float32)
    adata.obsm["X_quad_tgt"] = rng.normal(size=(n_obs, 6)).astype(np.float32)
    
    adata.write_h5ad(out)
    return out


def test_smoke_fit_cfm_panel(tiny_panel_adata, tmp_path):
    run_dir = tmp_path / "cfm_panel_smoke"
    
    result = subprocess.run(
        [
            sys.executable, "docs/scripts/fit_cfm.py",
            "--config-name", "fit_cfm_panel_smoke",
            f"paths.target_adata={tiny_panel_adata}",
            f"save.run_dir={run_dir}",
            "trainer.accelerator=cpu",
            "trainer.devices=1",
            "run.allow_overwrite=true",
        ],
        cwd=Path(__file__).resolve().parents[1],  # repo root
        capture_output=True,
        text=True,
        timeout=300,
    )
    
    # check exit code with log attached
    assert result.returncode == 0, (
        f"fit_cfm_panel_smoke exited {result.returncode}\n"
        f"--- stdout ---\n{result.stdout[-4000:]}\n"
        f"--- stderr ---\n{result.stderr[-4000:]}"
    )
    
    # artifacts from save_pretrained / save_transform
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "module" / "model.safetensors").is_file()
    assert (run_dir / "transform.json").is_file()
    assert (run_dir / "transform_params.npz").is_file()
    
    # the smoke config should have reused the pre-existing split
    assert "reusing the existing split and skipping the splitter" in result.stdout
