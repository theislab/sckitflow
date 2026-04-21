import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from sc_flow import SCFlow
from sc_flow.data._manager import DataManager
from sc_flow.methods._methods import BaseMethod


# -----------------------------------------------------------------------------
# Dummy module (picklable, no recursion)
# -----------------------------------------------------------------------------
class DummyModule:
    """Simple dummy module that mimics the interface but is picklable."""

    def forward(self, *args, **kwargs):
        pass

    def cpu(self):
        return self

    def to(self, device):
        return self

    def parameters(self):
        return []

    def train(self):
        pass

    def eval(self):
        pass


# -----------------------------------------------------------------------------
# Dummy PredictionData and Method for testing – no backend dependencies
# -----------------------------------------------------------------------------
class DummyPredictionData:
    def __init__(self, samples, traj=None):
        self.samples = samples
        self.traj = traj

    @classmethod
    def concatenate(cls, preds):
        samples = np.concatenate([p.samples for p in preds], axis=0)
        trajs = [p.traj for p in preds if p.traj is not None]
        traj = np.concatenate(trajs, axis=1) if trajs else None
        return cls(samples, traj)


class DummyMethod(BaseMethod):
    _module_cls = None

    def __init__(self, dims_registry, dm, is_paired_setting, *args, **kwargs):
        # Do NOT call super().__init__ because it tries to create a module.
        self._dims_registry = dims_registry
        self._dm = dm
        self._is_paired_setting = is_paired_setting
        self._module = DummyModule()  # picklable dummy
        self._train_mode = True

    def set_train_mode(self, mode: bool):
        self._train_mode = mode

    def train_step(self, *args, **kwargs):
        return {"loss": 0.0}

    def predict(self, matched_distr, *args, **kwargs):
        n_obs = len(matched_distr.target_distr.ann_df)
        n_feat = len(self._dims_registry.feature_names)
        # Deterministic output: zeros (for reproducible tests)
        samples = np.zeros((n_obs, n_feat))
        return DummyPredictionData(samples)


# -----------------------------------------------------------------------------
# Test suite
# -----------------------------------------------------------------------------
class TestSCFlow:
    """Test suite for SCFlow orchestration class."""

    # --------------------------------------------------------------------------
    # Registration and Initialization
    # --------------------------------------------------------------------------
    def test_register_adata_sets_class_attributes(self, adata: AnnData):
        SCFlow.register_adata(adata)
        assert isinstance(SCFlow._dm_cls, DataManager)
        assert SCFlow._dims_registry is not None
        assert SCFlow._is_paired_setting_cls is False
        assert SCFlow._dims_registry.feature_names is not None
        assert len(SCFlow._dims_registry.feature_names) == adata.n_vars

    def test_register_adata_with_control_key_sets_paired_true(self, adata: AnnData):
        SCFlow.register_adata(adata, control_values_dict={"drug": "control"})
        assert SCFlow._is_paired_setting_cls is True

    def test_init_copies_class_attrs_to_instance(self, adata: AnnData):
        SCFlow.register_adata(adata)
        model = SCFlow(method_cls=DummyMethod, backend="torch")
        assert model._dm is SCFlow._dm_cls
        assert model._dims_registry is SCFlow._dims_registry
        assert model._is_paired_setting is SCFlow._is_paired_setting_cls

    def test_init_raises_without_registration(self):
        SCFlow._dm_cls = None
        with pytest.raises(RuntimeError, match="Data has not been registered"):
            SCFlow(method_cls=DummyMethod)

    # --------------------------------------------------------------------------
    # Method Resolution and Backend Handling
    # --------------------------------------------------------------------------
    def test_method_id_resolves_to_registered_class(self, adata: AnnData, monkeypatch):
        mock_registry = {"cfm": DummyMethod}
        monkeypatch.setattr("sc_flow.backends.torch.methods.METHODS_REGISTRY", mock_registry)
        SCFlow.register_adata(adata)
        model = SCFlow(method_id="cfm", backend="torch")
        assert isinstance(model.method, DummyMethod)

    def test_unsupported_backend_raises_error(self, adata: AnnData, monkeypatch):
        # Mock the method registry so that 'cfm' exists (to avoid a KeyError)
        mock_registry = {"cfm": DummyMethod}
        monkeypatch.setattr("sc_flow.backends.torch.methods.METHODS_REGISTRY", mock_registry)
        SCFlow.register_adata(adata)
        # Using method_id forces backend validation (the registry lookup will fail because backend is invalid)
        with pytest.raises(RuntimeError, match="not supported"):
            SCFlow(method_id="cfm", backend="tensorflow")

    def test_backend_property(self, adata: AnnData):
        SCFlow.register_adata(adata)
        model = SCFlow(method_cls=DummyMethod, backend="torch")
        assert model.backend == "torch"

    # --------------------------------------------------------------------------
    # _to_numpy Conversion
    # --------------------------------------------------------------------------
    def test_to_numpy_with_torch_tensor(self, adata: AnnData):
        SCFlow.register_adata(adata)
        model = SCFlow(method_cls=DummyMethod, backend="torch")
        torch = pytest.importorskip("torch")
        t = torch.tensor([1.0, 2.0, 3.0])
        arr = model._to_numpy(t)
        assert isinstance(arr, np.ndarray)
        np.testing.assert_array_equal(arr, np.array([1.0, 2.0, 3.0]))

    def test_to_numpy_with_jax_array(self, adata: AnnData):
        SCFlow.register_adata(adata)
        model = SCFlow(method_cls=DummyMethod, backend="jax")
        jnp = pytest.importorskip("jax.numpy")
        arr = model._to_numpy(jnp.array([1.0, 2.0]))
        assert isinstance(arr, np.ndarray)

    def test_to_numpy_returns_none_for_none(self, adata: AnnData):
        SCFlow.register_adata(adata)
        model = SCFlow(method_cls=DummyMethod)
        assert model._to_numpy(None) is None

    # --------------------------------------------------------------------------
    # Training Orchestration (with mocking)
    # --------------------------------------------------------------------------
    def test_train_calls_trainer_and_sets_mode(self, adata: AnnData):
        SCFlow.register_adata(adata)
        model = SCFlow(method_cls=DummyMethod, backend="torch")
        with patch("sc_flow._model.Trainer") as mock_trainer_cls:
            mock_trainer = mock_trainer_cls.return_value
            set_train_mode_spy = MagicMock(wraps=model.method.set_train_mode)
            model.method.set_train_mode = set_train_mode_spy
            model.train(adata, n_train_steps=10, valid_freq=5, train_batch_size=32)
            mock_trainer_cls.assert_called_once_with(model.method)
            set_train_mode_spy.assert_called_once_with(True)
            mock_trainer.train.assert_called_once()
            call_kwargs = mock_trainer.train.call_args[1]
            assert call_kwargs["n_train_steps"] == 10
            assert call_kwargs["valid_freq"] == 5

    def test_train_with_validation_samplers(self, adata: AnnData):
        SCFlow.register_adata(adata)
        model = SCFlow(method_cls=DummyMethod)
        with patch("sc_flow._model.Trainer") as mock_trainer_cls:
            mock_trainer = mock_trainer_cls.return_value
            val_adatas = {"val1": adata, "val2": adata}
            model.train(adata, val_adatas_dict=val_adatas, n_train_steps=5)
            # The real Trainer.train is called as:
            # train(train_sampler, val_samplers_dict, n_train_steps=..., valid_freq=...)
            args, kwargs = mock_trainer.train.call_args
            # val_samplers_dict is the second positional argument
            assert len(args) >= 2
            val_samplers = args[1]
            assert isinstance(val_samplers, dict)
            assert set(val_samplers.keys()) == {"val1", "val2"}

    # --------------------------------------------------------------------------
    # Prediction Pipeline
    # --------------------------------------------------------------------------
    def test_predict_returns_anndata_with_correct_shape(self, adata: AnnData):
        SCFlow.register_adata(adata)
        model = SCFlow(method_cls=DummyMethod)
        pred_adata = model.predict(adata)
        assert isinstance(pred_adata, AnnData)
        assert pred_adata.n_obs == adata.n_obs
        assert pred_adata.n_vars == adata.n_vars
        pd.testing.assert_index_equal(pred_adata.obs.index, adata.obs.index)

    def test_predict_with_return_tensors(self, adata: AnnData):
        SCFlow.register_adata(adata)
        model = SCFlow(method_cls=DummyMethod)
        result = model.predict(adata, return_tensors=True)
        assert isinstance(result, tuple) and len(result) == 2
        pred_adata, pred_data = result
        assert isinstance(pred_adata, AnnData)
        assert hasattr(pred_data, "samples") and hasattr(pred_data, "traj")

    def test_predict_empty_tree_returns_empty_anndata(self, adata: AnnData, monkeypatch):
        SCFlow.register_adata(adata)
        model = SCFlow(method_cls=DummyMethod)
        mock_tree = MagicMock()
        mock_tree.flatten.return_value = ()
        monkeypatch.setattr(model._dm, "compile_adata", lambda x: mock_tree)
        pred_adata = model.predict(adata)
        assert pred_adata.n_obs == 0
        assert pred_adata.n_vars == adata.n_vars

    def test_predict_sets_eval_mode(self, adata: AnnData):
        SCFlow.register_adata(adata)
        model = SCFlow(method_cls=DummyMethod)
        set_train_mode_spy = MagicMock(wraps=model.method.set_train_mode)
        model.method.set_train_mode = set_train_mode_spy
        model.predict(adata)
        set_train_mode_spy.assert_called_once_with(False)

    # --------------------------------------------------------------------------
    # Properties
    # --------------------------------------------------------------------------
    def test_properties(self, adata: AnnData):
        SCFlow.register_adata(adata)
        model = SCFlow(method_cls=DummyMethod)
        assert isinstance(model.dm, DataManager)
        assert model.is_paired_setting is False
        assert isinstance(model.method, BaseMethod)
        assert model.trainer is None
        with patch("sc_flow._model.Trainer"):
            model.train(adata, n_train_steps=1)
        assert model.trainer is not None

    # --------------------------------------------------------------------------
    # Save / Load
    # --------------------------------------------------------------------------
    def test_save_load_and_predict(self, adata):
        SCFlow.register_adata(adata)
        model = SCFlow(method_cls=DummyMethod, backend="torch")
        pred1 = model.predict(adata)

        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
            tmp_path = tmp.name
        model.save(tmp_path, allow_overwrite=True)

        loaded = SCFlow.load(tmp_path, map_location="cpu")
        pred2 = loaded.predict(adata)

        np.testing.assert_array_equal(pred1.X, pred2.X)
        os.unlink(tmp_path)

    def test_save_load_and_continue_training(self, adata):
        SCFlow.register_adata(adata)
        model = SCFlow(method_cls=DummyMethod, backend="torch")
        model.train(adata, n_train_steps=5, train_batch_size=4)

        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
            tmp_path = tmp.name
        model.save(tmp_path, allow_overwrite=True)

        loaded = SCFlow.load(tmp_path, map_location="cpu")
        # Should not raise
        loaded.train(adata, n_train_steps=5, train_batch_size=4)

        os.unlink(tmp_path)
