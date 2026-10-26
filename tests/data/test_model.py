import json

import pandas as pd
import pytest
import torch
from anndata import AnnData
from scfit.pretrained import save_pretrained
from tests.data.shared import with_split

from sckitflow import Model, ModelConfig
from sckitflow.core.methods.inference._ode import ODEInferenceConfig
from sckitflow.core.methods.training._cfm import CFMTrainingConfig
from sckitflow.core.nn.composite.fields._base import NeuralVelocityConfig
from sckitflow.data._datamodule import FlowDataModule
from sckitflow.data._group_encoders import OneHotEncoderConfig
from sckitflow.data._manager import DataManagerConfig

from .conftest import CELL_LINES

SCHEMA = DataManagerConfig(
    categorical_conditions={"drug": ("drug",)},
    categorical_reps={"drug": "drug"},
    groups=("cell_line",),
    groups_encoding={"cell_line": OneHotEncoderConfig(categories=tuple(CELL_LINES))},
    split_by="split",
)
SPEC = ModelConfig(
    data=SCHEMA,
    module=NeuralVelocityConfig(),
    training=CFMTrainingConfig(),
    inference=ODEInferenceConfig(n_steps=5),
)


def _split(adata: AnnData) -> AnnData:
    """The experiment's split, written before the data reaches the model."""
    return with_split(adata, cols=("cell_line", "drug"), labels=("train", "test"), control_col="drug")


def test_model_round_trips(tmp_path, adata_small: AnnData):
    adata = _split(adata_small)
    model = SPEC.build(adata)
    model.save_pretrained(tmp_path)
    document = json.loads((tmp_path / "config.json").read_text())
    assert document["type"] == "model" and document["data"]["split_by"] == "split"
    assert document["data"]["groups_encoding"]["cell_line"]["type"] == "group_encoder.one_hot"

    loaded = Model.from_pretrained(tmp_path, adata)
    assert loaded.spec == SPEC
    for k, v in model.module.state_dict().items():
        assert loaded.module.state_dict()[k].equal(v)
    assert loaded.inference_method is not None


def test_checkpoint_state_is_plain_data(tmp_path, adata_small: AnnData):
    """The schema travels as a spec, so the state loads without unpickling."""
    adata = _split(adata_small)
    model = SPEC.build(adata)
    datamodule = FlowDataModule(model.manager, model.dims, adata, batch_size=4)
    assert "test" in datamodule.val_names
    torch.save(datamodule.state_dict(), tmp_path / "state.pt")

    restored = FlowDataModule(datamodule.dm, datamodule.data_dims)
    restored.load_state_dict(torch.load(tmp_path / "state.pt", weights_only=True))
    assert restored.dm.config == datamodule.dm.config
    assert restored.data_dims.feature_names.equals(datamodule.data_dims.feature_names)
    assert restored.data_dims.state_dim == datamodule.data_dims.state_dim


def test_from_pretrained_needs_a_model_with_module_weights(tmp_path, adata_small: AnnData):
    save_pretrained(tmp_path / "not_a_model", SCHEMA)
    with pytest.raises(TypeError, match="not a model"):
        Model.from_pretrained(tmp_path / "not_a_model", adata_small)
    save_pretrained(tmp_path / "no_weights", SPEC)
    with pytest.raises(ValueError, match="no module weights"):
        Model.from_pretrained(tmp_path / "no_weights", adata_small)


def test_reload_keeps_the_training_vocabulary(tmp_path, adata_small: AnnData):
    """A reloaded model encodes each group as in training, even when the new data lacks one of them."""
    first, *rest = CELL_LINES
    without_first = adata_small[adata_small.obs["cell_line"] != first].copy()

    def encodings(model: Model) -> dict:
        schema = model.manager.groups_data_schema
        return {
            line: schema.categorical_data(pd.DataFrame({"cell_line": [line]}), without_first.uns)
            .extract_reps()
            .mapping["cell_line"]
            .tolist()
            for line in rest
        }

    trained = SPEC.build(_split(adata_small))
    trained.save_pretrained(tmp_path)
    reloaded = Model.from_pretrained(tmp_path, _split(without_first))
    assert encodings(reloaded) == encodings(trained)
    assert reloaded.dims.groups_reps_dims == trained.dims.groups_reps_dims
