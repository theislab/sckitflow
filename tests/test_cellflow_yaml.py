"""The CellFlow YAML under ``docs/configs`` trains, saves and predicts with nothing but the file and the data."""

from pathlib import Path

import lightning as pl
import numpy as np
import pytest
import scipy.sparse as sp
import torch
import yaml
from anndata import AnnData

from sckitflow import Model, ModelConfig, predict_adata
from sckitflow.core.methods import MatchedTrainingMethod
from sckitflow.data import FlowDataModule
from sckitflow.data.sim import get_dummy_adata
from sckitflow.data.splitters import CombinationSplitterConfig
from sckitflow.trainer import TrainingPlan

CELLFLOW = Path(__file__).parents[1] / "docs" / "configs" / "cellflow.yaml"


@pytest.fixture(scope="module")
def adata() -> AnnData:
    rng = np.random.default_rng(0)
    adata = get_dummy_adata(
        n_obs_pert=600,
        n_obs_ctrl=200,
        n_genes=10,
        obsm_keys_to_dim={"X_pca": 8},
        obs_columns_to_nunique_and_prefix={"drug1": (3, "drug"), "drug2": (3, "drug"), "cell_line": (2, "cell_line")},
        uns_keys_to_nunique_prefix_and_dim={"drug": (3, "drug", 16), "cell_line": (2, "cell_line", 4)},
        obs_columns_to_fixed_val={"drug1": "control", "drug2": "control"},
        rng=rng,
    )
    adata.X = sp.csr_matrix(adata.X)  # real counts are sparse; OT must not read them
    adata.obsm["X_pca"] = rng.standard_normal((adata.n_obs, 8))  # the dummy obsm is all zeros, which OT cannot scale
    # the experiment's split, written before the data reaches the run (`split_by: split` in the YAML)
    splitter = CombinationSplitterConfig(
        group_keys=("cell_line", "drug1", "drug2"),
        always_train_keys=("cell_line",),
        control_key="drug1",
        test_fraction=0.2,
    )
    return splitter.build(rng=np.random.default_rng(0)).split(adata)


def _spec() -> ModelConfig:
    return ModelConfig.from_spec(yaml.safe_load(CELLFLOW.read_text()))


def test_cellflow_yaml_trains_saves_and_predicts(tmp_path, adata: AnnData):
    model = _spec().build(adata)
    # the experiment's side, set by a pipeline: a smaller run of the same model
    datamodule = FlowDataModule(model.manager, model.dims, adata, batch_size=32, n_train_steps=3)
    plan = TrainingPlan(
        model.training_method, torch.optim.Adam(model.module.parameters(), lr=1e-4), val_names=datamodule.val_names
    )
    assert isinstance(plan.training_method, MatchedTrainingMethod)
    before = {k: v.clone() for k, v in plan.module.state_dict().items()}
    matcher, calls = plan.training_method.matcher, []
    match_fn = matcher.match_fn
    matcher._match_fn = lambda **kw: calls.append(1) or match_fn(**kw)

    pl.Trainer(max_steps=3, accelerator="cpu", logger=False, enable_checkpointing=False, enable_progress_bar=False).fit(
        plan, datamodule=datamodule
    )
    assert len(calls) == 3  # every step is OT-matched
    assert any(not v.equal(before[k]) for k, v in plan.module.state_dict().items())

    model.save_pretrained(tmp_path)
    loaded = Model.from_pretrained(tmp_path, adata)
    for k, v in model.module.state_dict().items():
        assert loaded.module.state_dict()[k].equal(v)

    assert loaded.inference_method is not None
    pred = predict_adata(datamodule, loaded.inference_method, adata[~adata.obs["is_control"]].copy(), max_per_group=8)
    assert isinstance(pred, AnnData)
    assert pred.n_obs > 0
    assert torch.isfinite(torch.as_tensor(np.asarray(pred.X))).all()


def test_input_dim_comes_from_the_data():
    module = yaml.safe_load(CELLFLOW.read_text())["module"]
    module["condition_encoder_input_layers"]["drug"]["input_dim"] = 3
    with pytest.raises(ValueError, match="comes from the data"):
        ModelConfig.from_spec({**yaml.safe_load(CELLFLOW.read_text()), "module": module})
