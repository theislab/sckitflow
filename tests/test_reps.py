"""Reps are :mod:`anndata.acc` references, written in configs as anndata's dotted strings."""

from pathlib import Path

import lightning as pl
import numpy as np
import pytest
import torch
import yaml
from anndata import AnnData
from anndata.acc import A
from tests.data.shared import with_split

from sckitflow import Model, ModelConfig, predict_adata
from sckitflow.core.coupling import IndependentCouplingConfig
from sckitflow.core.methods import MatchedTrainingMethod, Matcher
from sckitflow.data import DataManagerConfig, FlowDataModule
from sckitflow.data.sim import get_dummy_adata
from sckitflow.trainer import TrainingPlan

CELLFLOW = Path(__file__).parents[1] / "docs" / "configs" / "cellflow.yaml"
REP_FIELDS = ("target_sample_rep", "source_sample_rep", "rep_lin", "source_rep_quad", "target_rep_quad")


class TestConfigStrings:
    @pytest.mark.parametrize(
        ("text", "ref"), [("X", A.X), ("obsm.X_pca", A.obsm["X_pca"]), ("layers.counts", A.layers["counts"])]
    )
    @pytest.mark.parametrize("field", REP_FIELDS)
    def test_strings_parse_to_references(self, field, text, ref):
        config = DataManagerConfig(**{"target_sample_rep": "X", field: text})
        assert getattr(config, field) == ref

    @pytest.mark.parametrize("field", REP_FIELDS)
    def test_a_bare_name_is_rejected(self, field):
        with pytest.raises(ValueError, match=r"'X_pca' does not say where the array lives: write 'obsm\.X_pca'"):
            DataManagerConfig(**{"target_sample_rep": "X", field: "X_pca"})

    @pytest.mark.parametrize("field", ["continuous_conditions", "target_continuous_covs"])
    def test_covariates_are_references_too(self, field):
        config = DataManagerConfig(target_sample_rep="X", **{field: ("obsm.dose", "layers.time")})
        assert getattr(config, field) == (A.obsm["dose"], A.layers["time"])
        with pytest.raises(ValueError, match=r"'dose' does not say where the array lives: write 'obsm\.dose'"):
            DataManagerConfig(target_sample_rep="X", **{field: ("dose",)})
        with pytest.raises(ValueError, match="A.X is not allowed here"):
            DataManagerConfig(target_sample_rep="X", **{field: ("X",)})
        with pytest.raises(ValueError, match=r"keyed by their array key, and \['dose'\] name several"):
            DataManagerConfig(target_sample_rep="X", **{field: ("obsm.dose", "layers.dose")})

    @pytest.mark.parametrize("bad", ["obsm.", "layers.", A.obsm["X_pca.v2"], A.layers["a.b"]])
    def test_keys_that_would_not_read_back_are_rejected(self, bad):
        with pytest.raises(ValueError, match="non-empty and hold no '.'"):
            DataManagerConfig(target_sample_rep=bad)

    def test_the_json_schema_types_reps_as_strings(self):
        props = DataManagerConfig.model_json_schema()["properties"]
        assert props["target_sample_rep"]["type"] == "string"

    @pytest.mark.parametrize("bad", ["obs.drug", "varm.PCs", A.var["gene"], 3])
    def test_only_whole_obs_arrays(self, bad):
        with pytest.raises(ValueError):
            DataManagerConfig(target_sample_rep=bad)

    def test_the_state_rep_is_required(self):
        with pytest.raises(ValueError, match="target_sample_rep"):
            DataManagerConfig()


def test_the_spec_round_trips_through_yaml():
    config = DataManagerConfig(
        target_sample_rep=A.layers["counts"],
        source_sample_rep="obsm.X_atac",
        rep_lin="X",
        source_rep_quad="obsm.X_src",
        target_rep_quad="layers.tgt",
        continuous_conditions=("obsm.dose",),
        target_continuous_covs=("layers.resp",),
    )
    spec = config.to_spec()
    assert {k: spec[k] for k in (*REP_FIELDS, "continuous_conditions", "target_continuous_covs")} == {
        "target_sample_rep": "layers.counts",
        "source_sample_rep": "obsm.X_atac",
        "rep_lin": "X",
        "source_rep_quad": "obsm.X_src",
        "target_rep_quad": "layers.tgt",
        "continuous_conditions": ["obsm.dose"],
        "target_continuous_covs": ["layers.resp"],
    }
    assert DataManagerConfig.from_spec(yaml.safe_load(yaml.safe_dump(spec))) == config


@pytest.fixture(scope="module")
def adata() -> AnnData:
    rng = np.random.default_rng(0)
    adata = get_dummy_adata(
        n_obs_pert=300,
        n_obs_ctrl=100,
        n_genes=10,
        obs_columns_to_nunique_and_prefix={"drug1": (3, "drug"), "drug2": (3, "drug"), "cell_line": (2, "cell_line")},
        uns_keys_to_nunique_prefix_and_dim={"drug": (3, "drug", 16), "cell_line": (2, "cell_line", 4)},
        obs_columns_to_fixed_val={"drug1": "control", "drug2": "control"},
        rng=rng,
    )
    adata.X = np.zeros_like(adata.X)  # a model that read .X would see nothing
    adata.layers["lognorm"] = rng.standard_normal(adata.shape).astype(np.float32)
    return with_split(
        adata, cols=("cell_line", "drug1", "drug2"), labels=("train",) * 4 + ("test",), control_col="drug1"
    )


def _cellflow(**data) -> ModelConfig:
    spec = yaml.safe_load(CELLFLOW.read_text())
    return ModelConfig.from_spec({**spec, "data": {**spec["data"], **data}})


def test_a_layer_is_the_state_and_the_coupling_rep(tmp_path, adata: AnnData):
    """OT-matched flow matching on a layer: the batches, the matcher and the prediction all read the layer."""
    model = _cellflow(target_sample_rep="layers.lognorm", rep_lin="layers.lognorm").build(adata)
    assert model.dims.state_dim == adata.n_vars
    datamodule = FlowDataModule(model.manager, model.dims, adata, batch_size=32, n_train_steps=3)
    batch = next(iter(datamodule.train_dataloader()))
    layer = adata.layers["lognorm"]
    for state in (batch["target_state"], batch["source_state"], batch["source_coupling_lin"]):
        assert state.abs().sum() > 0  # not the zeroed .X
        assert all(np.any(np.all(np.isclose(layer, r), axis=1)) for r in state.numpy())

    widths, match_fn = [], model.training_method.matcher.match_fn
    method = MatchedTrainingMethod(
        model.training_method.method,
        Matcher(lambda **kw: widths.append(kw["source_lin"].shape[1]) or match_fn(**kw)),
    )
    plan = TrainingPlan(method, torch.optim.Adam(model.module.parameters(), lr=1e-4), val_names=datamodule.val_names)
    pl.Trainer(max_steps=3, accelerator="cpu", logger=False, enable_checkpointing=False, enable_progress_bar=False).fit(
        plan, datamodule=datamodule
    )
    assert widths == [adata.n_vars] * 3

    model.save_pretrained(tmp_path)
    loaded = Model.from_pretrained(tmp_path, adata)
    assert loaded.spec == model.spec
    assert yaml.safe_load((tmp_path / "config.json").read_text())["data"]["target_sample_rep"] == "layers.lognorm"
    pred = predict_adata(datamodule, loaded.inference_method, adata[~adata.obs["is_control"]].copy(), max_per_group=4)
    assert list(pred.var_names) == list(adata.var_names)  # a layer keeps the genes' names
    assert torch.isfinite(torch.as_tensor(np.asarray(pred.X))).all()


def test_matching_without_a_coupling_rep_is_refused_at_build(adata: AnnData):
    """No `rep_lin`, no quadratics: OT has nothing to match on, and there is no silent fallback to .X."""
    with pytest.raises(ValueError, match="matches on rep_lin, but the data manager names no rep_lin"):
        _cellflow(target_sample_rep="layers.lognorm", rep_lin=None).build(adata)


def test_independent_coupling_needs_no_coupling_rep(adata: AnnData):
    """Random pairs read only the row counts, so they pair the states without a coupling rep."""
    spec = _cellflow(target_sample_rep="layers.lognorm", rep_lin=None).to_spec()
    spec["training"]["coupling"] = IndependentCouplingConfig().to_spec()
    model = ModelConfig.from_spec(spec).build(adata)
    datamodule = FlowDataModule(model.manager, model.dims, adata, batch_size=32, n_train_steps=1)
    batch = next(iter(datamodule.train_dataloader()))
    assert batch["source_coupling_lin"] is None
    loss, _ = model.training_method.compute_loss(batch, generator=torch.Generator().manual_seed(0))
    assert torch.isfinite(loss)
