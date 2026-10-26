"""The sources' own state rep (`source_sample_rep`): GENOT-style training from it, matched by Gromov-Wasserstein."""

import logging

import anndata as ad
import lightning as pl
import numpy as np
import pytest
import torch

from sckitflow import Model, ModelConfig, predict_adata
from sckitflow.core.coupling import GromovWassersteinCouplingConfig
from sckitflow.core.methods import CFMTrainingConfig, MatchedTrainingConfig, ODEInference, ODEInferenceConfig
from sckitflow.core.nn import NeuralVelocityConfig
from sckitflow.data import DataManagerConfig, FlowDataModule
from sckitflow.data._group_encoders import LabelEncoderConfig
from sckitflow.data.sim import get_dummy_adata
from sckitflow.trainer import TrainingPlan

ENCODER = {"source_encoder_mlp_kwargs": {"hidden_dims": [8]}, "source_encoder_output_dim": 8}


def _two_spaces(source_width: int = 4):
    """Controls whose state is `X_atac`, perturbed cells whose state is `X_rna` (6 wide); each file only has its own."""
    adata = get_dummy_adata(
        n_obs_pert=120,
        n_obs_ctrl=40,
        n_genes=10,
        obsm_keys_to_dim={},
        obs_columns_to_nunique_and_prefix={"drug": (3, "drug"), "cell_line": (2, "cell_line")},
        uns_keys_to_nunique_prefix_and_dim={"drug": (3, "drug", 4)},
        obs_columns_to_fixed_val={"drug": "control"},
        rng=np.random.default_rng(0),
    )
    adata.obs_names = adata.obs_names.astype(str)
    rng = np.random.default_rng(1)
    is_ctrl = (adata.obs["drug"] == "control").to_numpy()
    ctrl, pert = adata[is_ctrl].copy(), adata[~is_ctrl].copy()
    ctrl.obsm["X_atac"] = rng.normal(size=(ctrl.n_obs, source_width)).astype("float32")
    pert.obsm["X_rna"] = rng.normal(size=(pert.n_obs, 6)).astype("float32")
    return ctrl, pert


def _spec(*, generate_from_noise: bool = True, module: dict | None = None) -> ModelConfig:
    return ModelConfig(
        data=DataManagerConfig(
            target_sample_rep="X_rna",
            source_sample_rep="X_atac",
            categorical_conditions={"drug": ("drug",)},
            categorical_reps={"drug": "drug"},
            control_values_dict={"drug": "control"},
            groups=("cell_line",),
            groups_encoding={"cell_line": LabelEncoderConfig(classes=("cell_line0", "cell_line1"))},
            source_rep_quad="X_atac",
            target_rep_quad="X_rna",
            use_linear=False,
        ),
        module=NeuralVelocityConfig(**(ENCODER if module is None else module)),
        training=MatchedTrainingConfig(
            method=CFMTrainingConfig(generate_from_noise=generate_from_noise),
            # scaled costs: unscaled float32 squared distances make entropic GW underflow to a degenerate plan
            coupling=GromovWassersteinCouplingConfig(scale_cost="mean", epsilon=0.1, inner_stop_thr=1e-5),
        ),
        inference=ODEInferenceConfig(generate_from_noise=generate_from_noise, n_samples=1, n_steps=3),
    )


def test_generating_from_the_sources_own_rep(tmp_path, caplog):
    ctrl, pert = _two_spaces()
    model = _spec().build([ctrl, pert])
    assert (model.dims.state_dim, model.dims.source_state_dim) == (6, 4)
    datamodule = FlowDataModule(model.manager, model.dims, [ctrl, pert], batch_size=8, n_train_steps=3)
    plan = TrainingPlan(model.training_method, torch.optim.Adam(model.module.parameters(), lr=1e-3))
    with caplog.at_level(logging.WARNING):
        pl.Trainer(
            max_steps=3, accelerator="cpu", logger=False, enable_checkpointing=False, enable_progress_bar=False
        ).fit(plan, datamodule=datamodule)
    assert "uniform coupling" not in caplog.text  # the GW plan was real, not the uniform fallback

    model.save_pretrained(tmp_path)
    loaded = Model.from_pretrained(tmp_path, [ctrl, pert])
    for key, weight in model.module.state_dict().items():
        assert loaded.module.state_dict()[key].equal(weight)
    pred = predict_adata(datamodule, loaded.inference_method, [ctrl, pert], max_per_group=4)
    assert pred.n_vars == 6 and np.isfinite(np.asarray(pred.X)).all()  # predictions live in the targets' space


@pytest.mark.parametrize("aggregate", ["samples", "mean"])
def test_predictions_name_their_source_cells(aggregate):
    """Each prediction names the control cell it was conditioned on: its cell line's controls in obs order, tiled."""
    ctrl, pert = _two_spaces()
    model = _spec().build([ctrl, pert])
    datamodule = FlowDataModule(model.manager, model.dims, [ctrl, pert], batch_size=8, n_train_steps=3)
    inference = ODEInference(model.module, ODEInferenceConfig(generate_from_noise=True, n_samples=2), dims=model.dims)
    pred = predict_adata(datamodule, inference, [ctrl, pert], max_per_group=15, aggregate=aggregate)

    first = pred.obs if aggregate == "mean" else pred.obs[pred.obs["sample_idx"] == 0]
    assert len(first) == len(pred.obs) // (1 if aggregate == "mean" else 2)
    for (cell_line, _), obs in first.groupby(["cell_line", "drug"], observed=True):
        controls = ctrl.obs_names[ctrl.obs["cell_line"] == cell_line][:15]
        assert list(obs["source_obs_name"]) == [controls[i % len(controls)] for i in range(len(obs))]
    assert list(pred.var_names) == [f"X_rna_{i}" for i in range(6)]


def test_two_spaces_of_the_same_width_are_told_apart():
    ctrl, pert = _two_spaces(source_width=6)  # ATAC and RNA both 6 wide: only `source_sample_rep` tells them apart
    assert _spec().build([ctrl, pert]).dims.source_state_dim == 6
    with pytest.raises(ValueError, match=r"own state rep.*set `generate_from_noise=True`"):
        _spec(generate_from_noise=False).build([ctrl, pert])


def test_inference_alone_refuses_to_start_from_another_space():
    ctrl, pert = _two_spaces()
    model = _spec().build([ctrl, pert])
    with pytest.raises(ValueError, match="own state rep"):
        ODEInference(model.module, ODEInferenceConfig(generate_from_noise=False), dims=model.dims)


@pytest.mark.parametrize(
    ("module", "match"),
    [
        ({}, "only reads through a source encoder"),
        ({"source_encoder_mlp_kwargs": {"input_dim": 4}}, "comes from the data"),
    ],
)
def test_the_source_encoder_reads_the_other_space(module, match):
    ctrl, pert = _two_spaces()
    with pytest.raises(ValueError, match=match):
        _spec(module=module).build([ctrl, pert])


def test_reloading_needs_target_cells(tmp_path):
    ctrl, pert = _two_spaces()
    _spec().build([ctrl, pert]).save_pretrained(tmp_path)
    with pytest.raises(ValueError, match="None of the AnnData holds target cells"):
        Model.from_pretrained(tmp_path, ctrl)


def test_each_side_has_its_own_width_under_one_key():
    """The same obsm keys, holding a different width per side: state and quadratic term per side, a shared linear term."""
    n, d0, d1, shared, b = 100, 23, 28, 14, 8
    rng = np.random.default_rng(0)
    x0, x1 = rng.normal(size=(n, d0)), rng.normal(size=(n, d1))
    uns = {"source": {"control": np.array([1, 0]), "target": np.array([0, 1])}}
    adata0 = ad.AnnData(
        X=x0, obs={"source": "control"}, obsm={"X_rep": x0, "X_lin": x0[:, :shared], "X_quad": x0[:, shared:]}, uns=uns
    )
    adata1 = ad.AnnData(
        X=x1, obs={"source": "target"}, obsm={"X_rep": x1, "X_lin": x1[:, :shared], "X_quad": x1[:, shared:]}, uns=uns
    )
    dm = DataManagerConfig(
        categorical_conditions={"source": ["source"]},
        categorical_reps={"source": "source"},
        control_values_dict={"source": "control"},
        target_sample_rep="X_rep",
        source_sample_rep="X_rep",
        rep_lin="X_lin",
        source_rep_quad="X_quad",
        target_rep_quad="X_quad",
        use_linear=True,
    ).build()

    step_data = next(iter(dm.get_dataloaders([adata0, adata1], batch_size=b)["train"]))
    assert step_data["source_state"].shape == (b, d0)
    assert step_data["target_state"].shape == (b, d1)
    assert step_data["source_coupling_lin"].shape == (b, shared)
    assert step_data["target_coupling_lin"].shape == (b, shared)
    assert step_data["source_coupling_quad"].shape == (b, d0 - shared)
    assert step_data["target_coupling_quad"].shape == (b, d1 - shared)

    dims = dm.get_data_dimensionalities([adata0, adata1])  # and a model can be built on it
    assert (dims.state_dim, dims.source_state_dim) == (d1, d0)
    assert (dims.source_quad_dim, dims.target_quad_dim, dims.source_lin_dim) == (d0 - shared, d1 - shared, shared)
