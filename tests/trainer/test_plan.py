import copy

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import pytest
import torch
import torchmetrics
from anndata import AnnData

from sckitflow import predict_adata
from sckitflow.core._types import PredictionData
from sckitflow.core.methods import CFMTraining, ODEInference, ODEInferenceConfig
from sckitflow.core.nn import MLPVelocity
from sckitflow.data import DataManagerConfig, FlowDataModule
from sckitflow.data._group_encoders import OneHotEncoderConfig
from sckitflow.trainer import TrainingPlan

DRUGS = ("control", "a", "b", "c")
GROUPS = {"groups": ("cell_line",), "groups_encoding": {"cell_line": OneHotEncoderConfig()}}


def _adata() -> AnnData:
    """Two cell lines by four drugs, five cells each; drug `b` is `val1`, `c` is `val2`."""
    rng = np.random.default_rng(0)
    obs = pd.DataFrame({"cell_line": np.repeat(["c1", "c2"], 20), "drug": np.tile(np.repeat(DRUGS, 5), 2)})
    obs["split"] = obs["drug"].map({"control": "train", "a": "train", "b": "val1", "c": "val2"})
    adata = AnnData(rng.standard_normal((len(obs), 2)).astype(np.float32), obs=obs.astype("category"))
    adata.var_names = ["g0", "g1"]
    adata.uns["drug"] = {d: rng.standard_normal(3).astype(np.float32) for d in DRUGS}
    return adata


def _datamodule(adata: AnnData, **dm_kwargs) -> FlowDataModule:
    dm = DataManagerConfig(conditions={"drug": ("drug",)}, conditions_reps={"drug": "drug"}, **dm_kwargs).build()
    return FlowDataModule.from_adata(adata, dm, n_train_steps=10, batch_size=8)


def _module() -> MLPVelocity:
    return MLPVelocity(2, time_features_id="torch-cfm", vf_decoder_mlp_kwargs={"hidden_dims": [8]})


def _trainer(max_steps: int = 3, **kwargs) -> pl.Trainer:
    return pl.Trainer(
        max_steps=max_steps,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        **kwargs,
    )


def _inference(module: torch.nn.Module) -> ODEInference:
    return ODEInference(module, ODEInferenceConfig(n_steps=2))


def _fit(module: torch.nn.Module, seed: int) -> dict[str, torch.Tensor]:
    plan = TrainingPlan(CFMTraining(module), torch.optim.SGD(module.parameters(), lr=0.1), seed=seed)
    _trainer(limit_val_batches=0).fit(plan, datamodule=_datamodule(_adata(), split_by="split"))
    return {k: v.clone() for k, v in module.state_dict().items()}


def test_fit_logs_loss_and_updates_weights():
    module = _module()
    before = copy.deepcopy(module.state_dict())
    plan = TrainingPlan(CFMTraining(module), torch.optim.SGD(module.parameters(), lr=0.1))
    trainer = _trainer(limit_val_batches=0)
    trainer.fit(plan, datamodule=_datamodule(_adata(), split_by="split"))

    assert trainer.global_step == 3
    assert torch.isfinite(trainer.callback_metrics["loss"])
    assert any(not before[k].equal(v) for k, v in module.state_dict().items())


def _validate(val_names: tuple[str, ...]) -> dict[str, torch.Tensor]:
    module = _module()
    plan = TrainingPlan(
        CFMTraining(module),
        torch.optim.SGD(module.parameters(), lr=0.1),
        inference_method=_inference(module),
        metrics={"mse": torchmetrics.MeanSquaredError()},
        val_names=val_names,
    )
    dmod = _datamodule(_adata(), split_by="split")
    trainer = _trainer(max_steps=2, val_check_interval=2, check_val_every_n_epoch=None, num_sanity_val_steps=0)
    trainer.fit(plan, datamodule=dmod)
    return trainer.callback_metrics


def test_validation_logs_metrics():
    metrics = _validate(("val1", "val2"))
    assert torch.isfinite(metrics["val1/mse"])


def test_validation_logs_one_metric_per_val_name():
    metrics = _validate(("val1", "val2"))
    assert {"val1/mse", "val2/mse"} <= set(metrics)


def test_validation_without_inference_logs_nothing():
    module = _module()
    plan = TrainingPlan(
        CFMTraining(module), torch.optim.SGD(module.parameters()), metrics={"mse": torchmetrics.MeanSquaredError()}
    )
    trainer = _trainer(max_steps=2, val_check_interval=2, check_val_every_n_epoch=None, num_sanity_val_steps=0)
    trainer.fit(plan, datamodule=_datamodule(_adata(), split_by="split"))
    assert not any("/" in k for k in trainer.callback_metrics)


def test_configure_optimizers():
    module = _module()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    assert TrainingPlan(CFMTraining(module), optimizer).configure_optimizers() is optimizer

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    config = TrainingPlan(
        CFMTraining(module), optimizer, lr_scheduler=scheduler, lr_scheduler_interval="epoch"
    ).configure_optimizers()
    assert config == {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

    with pytest.raises(ValueError, match="without an optimizer"):
        TrainingPlan(CFMTraining(module)).configure_optimizers()


def test_same_seed_same_weights():
    module = _module()
    first = _fit(copy.deepcopy(module), seed=1)
    same = _fit(copy.deepcopy(module), seed=1)
    other = _fit(copy.deepcopy(module), seed=2)
    assert all(first[k].equal(same[k]) for k in first)
    assert any(not first[k].equal(other[k]) for k in first)


@pytest.fixture
def paired():
    """A datamodule with `control` registered as the drug's control value."""
    adata = _adata()
    return _datamodule(adata, **GROUPS, control_values_dict={"drug": "control"}), adata


def test_predict_adata_schema(paired):
    dmod, adata = paired
    pred = predict_adata(dmod, _inference(_module()), adata)
    assert isinstance(pred, AnnData)
    assert list(pred.var_names) == list(dmod.data_dims.feature_names) == ["g0", "g1"]
    assert {"cell_line", "drug"} <= set(pred.obs.columns)
    assert set(pred.obs["drug"]) == {"a", "b", "c"}
    assert pred.n_obs == 30


def test_predict_adata_return_raw(paired):
    dmod, adata = paired
    pred, raw = predict_adata(dmod, _inference(_module()), adata, return_raw=True)
    assert isinstance(raw, PredictionData)
    np.testing.assert_array_equal(np.asarray(raw.X), pred.X)


def test_predict_adata_max_per_group(paired):
    dmod, adata = paired
    pred = predict_adata(dmod, _inference(_module()), adata, max_per_group=2)
    assert pred.obs.value_counts(["cell_line", "drug"]).max() <= 2
    assert pred.n_obs == 2 * 6


@pytest.mark.parametrize(
    ("registered", "control_values_dict", "expected_drugs"),
    [
        ({"drug": "control"}, None, {"a", "b", "c"}),
        ({"drug": "control"}, {}, set(DRUGS)),
        (None, {"drug": "control"}, {"a", "b", "c"}),
        (None, {"drug": "nonexistent"}, set(DRUGS)),
    ],
)
def test_predict_adata_control_values(registered, control_values_dict, expected_drugs):
    adata = _adata()
    dmod = _datamodule(adata, **GROUPS, control_values_dict=registered)
    pred = predict_adata(dmod, _inference(_module()), adata, control_values_dict=control_values_dict)
    assert set(pred.obs["drug"]) == expected_drugs


@pytest.mark.parametrize("training", [True, False])
def test_predict_adata_restores_mode(paired, training):
    dmod, adata = paired
    module = _module().train(training)
    predict_adata(dmod, _inference(module), adata)
    assert module.training is training


def test_checkpoint_restores_datamodule(tmp_path):
    adata = _adata()
    dmod = _datamodule(adata, split_by="split")
    module = _module()
    trainer = _trainer(max_steps=1, limit_val_batches=0)
    trainer.fit(TrainingPlan(CFMTraining(module), torch.optim.SGD(module.parameters())), datamodule=dmod)
    path = tmp_path / "run.ckpt"
    trainer.save_checkpoint(path)

    restored = FlowDataModule.load_from_checkpoint(str(path), adata)
    assert restored.dm.config == dmod.dm.config
    assert restored.data_dims.feature_names.equals(dmod.data_dims.feature_names)
    assert restored.adata is adata
    assert restored.val_names == dmod.val_names
