import pytest

pytest.importorskip("torch")
pytest.importorskip("lightning")

import lightning.pytorch as pl

from sc_flow import SCFlow
from sc_flow.dataset.toy_data import get_toy_dataset


@pytest.fixture
def adata():
    return get_toy_dataset("moons", noise=0.1).adata


def _cfg(framework):
    return {
        "data": {"datamanager": {}},
        "method": {
            "method_id": "cfm",
            "backend": "torch",
            "config": {
                "velocity_field": {
                    "vf_decoder_mlp_kwargs": {"hidden_dims": [16, 16]},
                    "time_features_id": "torch-cfm",
                    "conditioning_id": "resnet1d",
                },
                "probability_path": {"kind": "linear-dirac"},
                "flow_solver": {"kind": "ode", "scheme": "euler", "num_steps": 10},
            },
        },
        "optim": {"optimizer_cls": "Adam", "lr": 1e-4},
        "trainer": {"framework": framework, "device": "cpu", "n_train_steps": 3, "train_batch_size": 32},
    }


class TestLightningBackend:
    def test_fit_runs_via_lightning(self, adata):
        model = SCFlow.from_config(_cfg("lightning"), adata)
        model.fit(adata)
        assert isinstance(model._lightning_trainer, pl.Trainer)
        assert model._lightning_trainer.global_step > 0

    def test_predict_after_lightning_fit(self, adata):
        model = SCFlow.from_config(_cfg("lightning"), adata)
        model.fit(adata)
        pred = model.predict(adata)
        assert pred.X.shape[1] == adata.X.shape[1]

    def test_native_still_default(self, adata):
        model = SCFlow.from_config(_cfg("native"), adata)
        model.fit(adata)
        assert model._lightning_trainer is None
        assert model.trainer is not None
