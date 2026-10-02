import json

import torch
from anndata import AnnData

from sckitflow import RunConfig, load_run, save_run
from sckitflow.core.methods.inference._ode import ODEInferenceConfig
from sckitflow.core.methods.training._cfm import CFMTrainingConfig
from sckitflow.data._config import FlowDataModuleConfig
from sckitflow.data._datamodule import FlowDataModule
from sckitflow.data._group_encoders import OneHotEncoderConfig
from sckitflow.data._manager import DataManagerConfig
from sckitflow.data.splitters._combination import CombinationSplitterConfig

SPLITTER = CombinationSplitterConfig(
    group_keys=("cell_line", "drug"), always_train_keys=("cell_line",), control_key="drug", test_fraction=0.5
)
DATA = FlowDataModuleConfig(
    manager=DataManagerConfig(
        conditions={"drug": ("drug",)},
        conditions_reps={"drug": "drug"},
        groups=("cell_line",),
        groups_encoding={"cell_line": OneHotEncoderConfig()},
    ),
    batch_size=4,
)


def test_load_run_round_trips(tmp_path, adata_small: AnnData):
    module = torch.nn.Linear(2, 2)
    spec = RunConfig(
        data=DATA,
        training=CFMTrainingConfig(),
        inference=ODEInferenceConfig(n_steps=5),
        splitter=SPLITTER,
        splitter_seed=3,
    )
    save_run(tmp_path, spec, module)
    document = json.loads((tmp_path / "specs.json").read_text())
    assert (document["loader_seed"], document["splitter_seed"]) == (0, 3)
    assert document["data"]["manager"]["groups_encoding"]["cell_line"]["type"] == "group_encoder.one_hot"

    datamodule, plan = load_run(tmp_path, adata_small, torch.nn.Linear(2, 2))
    assert plan.training_method.module.weight.equal(module.weight)
    assert "test" in datamodule.val_names


def test_checkpoint_state_is_plain_data(tmp_path, adata_small: AnnData):
    """The schema and the splitter travel as specs, so the state loads without unpickling."""
    run = RunConfig(data=DATA, training=CFMTrainingConfig(), splitter=SPLITTER, splitter_seed=3)
    datamodule = run.build(adata_small, torch.nn.Linear(2, 2)).datamodule
    torch.save(datamodule.state_dict(), tmp_path / "state.pt")

    restored = FlowDataModule(datamodule.dm, datamodule.data_dims)
    restored.load_state_dict(torch.load(tmp_path / "state.pt", weights_only=True))
    assert restored.dm.config == datamodule.dm.config
    assert restored.data_dims.feature_names.equals(datamodule.data_dims.feature_names)
    assert restored.data_dims.state_dim == datamodule.data_dims.state_dim
    assert restored.dm.splitter.assign(adata_small).equals(datamodule.dm.splitter.assign(adata_small))
