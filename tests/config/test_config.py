import pytest

pytest.importorskip("torch")

from omegaconf import OmegaConf

from sc_flow import SCFlow
from sc_flow.backends.torch.methods import CFM
from sc_flow.config import RunConfig
from sc_flow.dataset.toy_data import get_toy_dataset


@pytest.fixture
def adata():
    return get_toy_dataset("moons", noise=0.1).adata


def _cfg(**trainer):
    return {
        "data": {"datamanager": {}},
        "method": {
            "method_id": "cfm",
            "config": {
                "velocity_field": {
                    "vf_decoder_mlp_kwargs": {"hidden_dims": [16, 16]},
                    "time_features_id": "torch-cfm",
                    "conditioning_id": "resnet1d",
                },
                "probability_path": {"kind": "linear-dirac"},
                "flow_solver": {"kind": "ode", "scheme": "euler", "num_steps": 20},
            },
        },
        "optim": {"lr": 1e-4},
        "trainer": {"device": "cpu", "n_train_steps": 2, "train_batch_size": 32, **trainer},
    }


class TestFromConfig:
    def test_builds_torch_cfm(self, adata):
        model = SCFlow.from_config(_cfg(), adata)
        assert isinstance(model._method, CFM)

    def test_fit_and_predict(self, adata):
        model = SCFlow.from_config(_cfg(), adata)
        model.fit(adata)
        pred = model.predict(adata)
        assert pred.X.shape[1] == adata.X.shape[1]

    def test_yaml_round_trip(self, tmp_path, adata):
        cfg_node = OmegaConf.merge(OmegaConf.structured(RunConfig), OmegaConf.create(_cfg()))
        path = tmp_path / "run.yaml"
        path.write_text(OmegaConf.to_yaml(cfg_node))
        model = SCFlow.from_yaml(str(path), adata)
        assert isinstance(model._method, CFM)


class TestValidation:
    def test_unsupported_device_raises(self, adata):
        with pytest.raises(ValueError, match=r"Device 'tpu' is not supported"):
            SCFlow.from_config(_cfg(device="tpu"), adata)

    def test_unknown_method_raises(self, adata):
        cfg = _cfg()
        cfg["method"]["method_id"] = "does-not-exist"
        with pytest.raises(KeyError, match=r"not found"):
            SCFlow.from_config(cfg, adata)


class TestCallbacksMetrics:
    def test_resolve_metrics(self):
        from sc_flow.config import resolve_metrics

        metrics = resolve_metrics({"mmd": {"kind": "mmd", "gammas": [1.0, 0.5]}}, "torch")
        assert "mmd" in metrics
        assert type(metrics["mmd"]).__name__ == "MaximumMeanDiscrepancy"

    def test_resolve_callbacks_wraps_metrics(self):
        from sc_flow.config import resolve_callbacks
        from sc_flow.config._run import TrainerConfig
        from sc_flow.trainer._callbacks import MetricsCallback

        tr = TrainerConfig(device="cpu", metrics={"mmd": {"kind": "mmd", "gammas": [1.0]}})
        cbs = resolve_callbacks(tr, "torch")
        assert len(cbs) == 1
        assert isinstance(cbs[0], MetricsCallback)

    def test_fit_with_configured_metrics_runs(self, adata):
        cfg = _cfg()
        cfg["trainer"]["metrics"] = {"mmd": {"kind": "mmd", "gammas": [1.0]}}
        model = SCFlow.from_config(cfg, adata)
        model.fit(adata)  # metrics callback built and threaded through the native loop
        assert model.trainer is not None


class TestCapabilities:
    def test_cfm_capabilities(self):
        caps = CFM.capabilities()
        assert caps.category == "flow"
        assert "mps" in caps.supported_devices
        assert caps.config_cls is not None
