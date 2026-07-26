import json

import anndata as ad
import numpy as np
import torch
from scipy import sparse

from sc_flow.concept import ContrastiveObjective, FoundationModel
from sc_flow.training import TrainingModule


def _adata(n: int = 64, g: int = 80, seed: int = 0) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    x = rng.poisson(0.5, size=(n, g)).astype(np.float32)
    a = ad.AnnData(X=sparse.csr_matrix(x))
    a.var_names = [f"ENSG{i:011d}" for i in range(g)]
    a.obs_names = [str(i) for i in range(n)]
    return a


def _recipe(adata: ad.AnnData) -> dict:
    return {
        "data": {"adata": adata},
        "backbone": {"dim_model": 16, "n_layers": 1, "n_heads": 2, "dim_feedforward": 32, "max_rank": 32},
        "objective": {},
        "sampler": {"batch_size": 16, "max_tokens": 24},
        "trainer": {"lr": 1e-3},
    }


def test_builds_lightning_pieces():
    fm = FoundationModel(_recipe(_adata()))
    assert isinstance(fm.module, TrainingModule)
    assert fm.callbacks == []
    assert fm.vocab.n_tokens == 80 + 2  # genes + PAD/CLS


def test_datamodule_schema_and_forward():
    fm = FoundationModel(_recipe(_adata()))
    batch = next(iter(fm.datamodule.train_dataloader()))
    assert set(batch) == {"tokens_1", "tokens_2", "pad_mask_1", "pad_mask_2"}
    assert batch["tokens_1"].shape[0] == 16
    # drive the objective directly (self.log needs a live Trainer, so we don't call training_step)
    loss, logs = ContrastiveObjective().compute_loss(fm.module.model, batch)
    assert torch.isfinite(loss)
    assert "retrieval_acc" in logs


def test_save_bundle_writes_specs(tmp_path):
    fm = FoundationModel(_recipe(_adata()))
    fm.save(tmp_path)
    cfg = json.loads((tmp_path / "config.json").read_text())
    assert cfg["backbone"]["type"] == "sc_flow.gene_encoder"
    assert cfg["objective"]["type"] == "sc_flow.concept_clip"
    assert cfg["family"] == "foundation"
    assert len(cfg["vocab_genes"]) == 80
    assert (tmp_path / "model.safetensors").exists()


def test_unimplemented_task_is_loud():
    import pytest

    r = _recipe(_adata())
    r["task"] = "classify"
    with pytest.raises(NotImplementedError, match="classify"):
        FoundationModel(r)
