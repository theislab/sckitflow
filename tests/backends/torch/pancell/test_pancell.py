import json

import anndata as ad
import numpy as np
import torch
from scipy import sparse

from sc_flow._registry import parse
from sc_flow.concept import GeneEncoderConfig
from sc_flow.families import available_families, build_family
from sc_flow.pancell import LinearFMObjectiveConfig, PanCellFlowModel, VelocityMLPConfig


def test_families_registered_on_import():
    assert set(available_families()) >= {"foundation", "pancell"}


def test_flow_objective_is_a_component_with_nested_path():
    cfg = LinearFMObjectiveConfig()
    spec = cfg.to_spec()
    assert spec["type"] == "flow.fm_linear"
    assert spec["config"]["probability_path"]["type"] == "flow.path.linear"  # nested Component
    assert parse(spec) == cfg


def test_velocity_config_roundtrips_and_forwards():
    cfg = VelocityMLPConfig(dim=16, hidden=32, n_layers=2)
    assert parse(cfg.to_spec()) == cfg
    v = cfg.build()
    assert v(torch.randn(4, 16), torch.rand(4, 1)).shape == (4, 16)


def _adata(n: int = 40, g: int = 60) -> ad.AnnData:
    rng = np.random.default_rng(0)
    a = ad.AnnData(X=sparse.csr_matrix(rng.poisson(0.5, size=(n, g)).astype(np.float32)))
    a.var_names = [f"ENSG{i:011d}" for i in range(g)]
    a.obs_names = [str(i) for i in range(n)]
    a.obs["is_control"] = [True] * (n // 2) + [False] * (n - n // 2)
    return a


def _recipe(adata):
    return {
        "data": {"adata": adata},
        "state_encoder": {"dim_model": 16, "n_layers": 1, "n_heads": 2, "dim_feedforward": 32, "max_rank": 24},
        "velocity": {"hidden": 32, "n_layers": 2},
        "sampler": {"batch_size": 8, "max_tokens": 16, "steps_per_epoch": 5},
        "trainer": {"lr": 1e-3},
    }


def test_pancell_builds_and_forwards():
    b = build_family("pancell", _recipe(_adata()))  # generic family dispatch
    assert isinstance(b.model, PanCellFlowModel)
    batch = next(iter(b.datamodule.train_dataloader()))
    assert set(batch) == {"source_tokens", "source_mask", "target_tokens", "target_mask"}
    loss, logs = LinearFMObjectiveConfig().build().compute_loss(b.model, batch)
    assert torch.isfinite(loss)
    assert "fm_loss" in logs


def test_frozen_state_encoder_excludes_params_from_grad():
    r = _recipe(_adata())
    r["state_encoder"]["freeze"] = True
    b = build_family("pancell", r)
    enc_grad = any(p.requires_grad for p in b.model.state_encoder.parameters())
    vel_grad = all(p.requires_grad for p in b.model.velocity.parameters())
    assert not enc_grad and vel_grad  # frozen encoder, trainable velocity


def test_pancell_save_records_component_specs(tmp_path):
    b = build_family("pancell", _recipe(_adata()))
    b.save(tmp_path)
    cfg = json.loads((tmp_path / "config.json").read_text())
    assert cfg["family"] == "pancell"
    assert cfg["state_encoder"]["type"] == "sc_flow.gene_encoder"
    assert cfg["velocity"]["type"] == "flow.velocity_mlp"
    assert cfg["objective"]["type"] == "flow.fm_linear"
    assert (tmp_path / "model.safetensors").exists()
