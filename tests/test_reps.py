"""Reps are :mod:`anndata.acc` references, written in configs as anndata's dotted strings."""

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from anndata import AnnData
from anndata.acc import A
from tests.data.shared import with_split

from sckitflow import ModelConfig
from sckitflow.core.coupling import IndependentCouplingConfig
from sckitflow.data import DataManagerConfig, FlowDataModule
from sckitflow.data.sim import get_dummy_adata

CELLFLOW = Path(__file__).parents[1] / "docs" / "configs" / "cellflow.yaml"


@pytest.mark.parametrize(
    ("text", "ref"), [("X", A.X), ("obsm.X_pca", A.obsm["X_pca"]), ("layers.counts", A.layers["counts"])]
)
def test_strings_parse_to_references(text, ref):
    assert DataManagerConfig(target_sample_rep=text).target_sample_rep == ref


@pytest.mark.parametrize("bad", ["X_pca", "obs.drug", "varm.PCs", A.var["gene"], 3])
def test_only_whole_obs_arrays(bad):
    with pytest.raises(ValueError):
        DataManagerConfig(target_sample_rep=bad)


def test_covariates_are_references_too():
    config = DataManagerConfig(target_sample_rep="X", continuous_conditions=("obsm.dose", "layers.time", "X"))
    assert config.continuous_conditions == (A.obsm["dose"], A.layers["time"], A.X)
    with pytest.raises(ValueError, match=r"\['dose'\] name several"):
        DataManagerConfig(target_sample_rep="X", continuous_conditions=("obsm.dose", "layers.dose"))


def test_the_state_rep_is_required():
    with pytest.raises(ValueError, match="target_sample_rep"):
        DataManagerConfig()


def test_the_spec_round_trips_through_yaml():
    config = DataManagerConfig(
        target_sample_rep=A.layers["counts"],
        rep_lin="X",
        source_rep_quad="obsm.X_src",
        continuous_conditions=("obsm.dose",),
    )
    spec = config.to_spec()
    assert (spec["target_sample_rep"], spec["rep_lin"], spec["source_rep_quad"]) == ("layers.counts", "X", "obsm.X_src")
    assert spec["continuous_conditions"] == ["obsm.dose"]
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
    adata.layers["lognorm"] = rng.standard_normal(adata.shape).astype(np.float32)
    return with_split(
        adata, cols=("cell_line", "drug1", "drug2"), labels=("train",) * 4 + ("test",), control_col="drug1"
    )


def _cellflow(**data) -> ModelConfig:
    spec = yaml.safe_load(CELLFLOW.read_text())
    return ModelConfig.from_spec({**spec, "data": {**spec["data"], **data}})


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
