# tests/test_scflow.py
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from sc_flow import SCFlow
from sc_flow.data._manager import DataManager
from sc_flow.methods._methods import BaseMethod


# -----------------------------------------------------------------------------
# Dummy PredictionData and Method for testing – no backend dependencies
# -----------------------------------------------------------------------------
class DummyPredictionData:
    """Fake PredictionData that works with numpy arrays."""

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
    """
    A completely mocked method that does not create any real modules.
    It bypasses super().__init__ to avoid backend initialisation.
    """

    _module_cls = None

    def __init__(self, dims_registry, dm, is_paired_setting, *args, **kwargs):
        # Do NOT call super().__init__ because it tries to create a module.
        self._dims_registry = dims_registry
        self._dm = dm
        self._is_paired_setting = is_paired_setting
        self._module = MagicMock()  # dummy module
        self._train_mode = True

    def set_train_mode(self, mode: bool):
        self._train_mode = mode

    def train_step(self, *args, **kwargs):
        return {"loss": 0.0}

    def predict(self, matched_distr, *args, **kwargs):
        n_obs = len(matched_distr.target_distr.ann_df)
        n_feat = len(self._dims_registry.feature_names)
        samples = np.random.randn(n_obs, n_feat)
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
        """Test that register_adata correctly initialises class-level state."""
        SCFlow.register_adata(adata)

        assert isinstance(SCFlow._dm_cls, DataManager)
        assert SCFlow._dims_registry is not None
        # No control_key was passed, so is_paired_setting should be False
        assert SCFlow._is_paired_setting_cls is False

        # Feature names should be captured
        assert SCFlow._dims_registry.feature_names is not None
        assert len(SCFlow._dims_registry.feature_names) == adata.n_vars

    def test_register_adata_with_control_key_sets_paired_true(self, adata: AnnData):
        """If a control_key is given, is_paired_setting becomes True."""
        SCFlow.register_adata(adata, control_values_dict={"drug": "control"})
        assert SCFlow._is_paired_setting_cls is True

    def test_init_copies_class_attrs_to_instance(self, adata: AnnData):
        """Instance attributes should reflect the registered class attributes."""
        SCFlow.register_adata(adata)
        model = SCFlow(method_cls=DummyMethod, backend="torch")

        assert model._dm is SCFlow._dm_cls
        assert model._dims_registry is SCFlow._dims_registry
        assert model._is_paired_setting is SCFlow._is_paired_setting_cls

    def test_init_raises_without_registration(self):
        """Creating a model without prior registration raises RuntimeError."""
        SCFlow._dm_cls = None  # Simulate unregistered state
        with pytest.raises(RuntimeError, match="Data has not been registered"):
            SCFlow(method_cls=DummyMethod)

    # --------------------------------------------------------------------------
    # Method Resolution and Backend Handling
    # --------------------------------------------------------------------------
    def test_method_id_resolves_to_registered_class(self, adata: AnnData, monkeypatch):
        """method_id should select the correct class from the backend registry."""
        mock_registry = {"cfm": DummyMethod}
        monkeypatch.setattr("sc_flow.backends.torch.methods.METHODS_REGISTRY", mock_registry)

        SCFlow.register_adata(adata)
        model = SCFlow(method_id="cfm", backend="torch")
        assert isinstance(model.method, DummyMethod)

    def test_invalid_method_id_raises_keyerror(self, adata: AnnData, monkeypatch):
        """Unknown method_id raises KeyError."""
        monkeypatch.setattr("sc_flow.backends.torch.methods.METHODS_REGISTRY", {})
        SCFlow.register_adata(adata)

        with pytest.raises(KeyError, match="not supported"):
            SCFlow(method_id="nonexistent", backend="torch")

    def test_unsupported_backend_raises_error(self, adata: AnnData):
        """Invalid backend string raises RuntimeError."""
        SCFlow.register_adata(adata)
        with pytest.raises(RuntimeError, match="not supported"):
            SCFlow(method_cls=DummyMethod, backend="tensorflow")

    def test_backend_property(self, adata: AnnData):
        """The backend property returns the value passed at init."""
        SCFlow.register_adata(adata)
        model = SCFlow(method_cls=DummyMethod, backend="torch")
        assert model.backend == "torch"

    # --------------------------------------------------------------------------
    # _to_numpy Conversion
    # --------------------------------------------------------------------------
    def test_to_numpy_with_torch_tensor(self, adata: AnnData):
        """PyTorch tensors are correctly converted to numpy arrays."""
        SCFlow.register_adata(adata)
        model = SCFlow(method_cls=DummyMethod, backend="torch")

        torch = pytest.importorskip("torch")
        t = torch.tensor([1.0, 2.0, 3.0])
        arr = model._to_numpy(t)

        assert isinstance(arr, np.ndarray)
        np.testing.assert_array_equal(arr, np.array([1.0, 2.0, 3.0]))

    def test_to_numpy_with_jax_array(self, adata: AnnData):
        """JAX arrays are correctly converted to numpy arrays."""
        SCFlow.register_adata(adata)
        model = SCFlow(method_cls=DummyMethod, backend="jax")

        jnp = pytest.importorskip("jax.numpy")
        arr = model._to_numpy(jnp.array([1.0, 2.0]))
        assert isinstance(arr, np.ndarray)

    def test_to_numpy_returns_none_for_none(self, adata: AnnData):
        """None input returns None."""
        SCFlow.register_adata(adata)
        model = SCFlow(method_cls=DummyMethod)
        assert model._to_numpy(None) is None

    # --------------------------------------------------------------------------
    # Training Orchestration (with mocking)
    # --------------------------------------------------------------------------
    def test_train_calls_trainer_and_sets_mode(self, adata: AnnData):
        """train() sets up samplers, trainer, and invokes training."""
        SCFlow.register_adata(adata)
        model = SCFlow(method_cls=DummyMethod, backend="torch")

        with patch("sc_flow.trainer._trainer.Trainer") as mock_trainer_cls:
            mock_trainer = mock_trainer_cls.return_value

            # Spy on set_train_mode
            set_train_mode_spy = MagicMock(wraps=model.method.set_train_mode)
            model.method.set_train_mode = set_train_mode_spy

            model.train(
                adata,
                n_train_steps=10,
                valid_freq=5,
                train_batch_size=32,
            )

            mock_trainer_cls.assert_called_once_with(model.method)
            set_train_mode_spy.assert_called_once_with(True)

            mock_trainer.train.assert_called_once()
            call_kwargs = mock_trainer.train.call_args[1]
            assert call_kwargs["n_train_steps"] == 10
            assert call_kwargs["valid_freq"] == 5

    def test_train_with_validation_samplers(self, adata: AnnData):
        """Validation samplers are created when val_adatas_dict is provided."""
        SCFlow.register_adata(adata)
        model = SCFlow(method_cls=DummyMethod)

        with patch("sc_flow.trainer._trainer.Trainer") as mock_trainer_cls:
            mock_trainer = mock_trainer_cls.return_value

            val_adatas = {"val1": adata, "val2": adata}
            model.train(adata, val_adatas_dict=val_adatas, n_train_steps=5)

            call_kwargs = mock_trainer.train.call_args[1]
            val_samplers = call_kwargs["val_samplers_dict"]
            assert isinstance(val_samplers, dict)
            assert set(val_samplers.keys()) == {"val1", "val2"}

    # --------------------------------------------------------------------------
    # Prediction Pipeline
    # --------------------------------------------------------------------------
    def test_predict_returns_anndata_with_correct_shape(self, adata: AnnData):
        """predict() returns an AnnData with the same number of cells."""
        SCFlow.register_adata(adata)
        model = SCFlow(method_cls=DummyMethod)

        pred_adata = model.predict(adata)

        assert isinstance(pred_adata, AnnData)
        assert pred_adata.n_obs == adata.n_obs
        assert pred_adata.n_vars == adata.n_vars
        pd.testing.assert_index_equal(pred_adata.obs.index, adata.obs.index)

    def test_predict_with_return_tensors(self, adata: AnnData):
        """return_tensors=True returns (AnnData, PredictionData)."""
        SCFlow.register_adata(adata)
        model = SCFlow(method_cls=DummyMethod)

        result = model.predict(adata, return_tensors=True)

        assert isinstance(result, tuple)
        assert len(result) == 2
        pred_adata, pred_data = result
        assert isinstance(pred_adata, AnnData)
        assert hasattr(pred_data, "samples")
        assert hasattr(pred_data, "traj")

    def test_predict_empty_tree_returns_empty_anndata(self, adata: AnnData, monkeypatch):
        """When the tree flattens to empty, an empty AnnData is returned."""
        SCFlow.register_adata(adata)
        model = SCFlow(method_cls=DummyMethod)

        mock_tree = MagicMock()
        mock_tree.flatten.return_value = ()
        monkeypatch.setattr(model._dm, "compile_adata", lambda x: mock_tree)

        pred_adata = model.predict(adata)
        assert pred_adata.n_obs == 0
        assert pred_adata.n_vars == adata.n_vars

    def test_predict_sets_eval_mode(self, adata: AnnData):
        """predict() sets the method to evaluation mode."""
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
        """Public properties return the expected objects."""
        SCFlow.register_adata(adata)
        model = SCFlow(method_cls=DummyMethod)

        assert isinstance(model.dm, DataManager)
        assert model.is_paired_setting is False
        assert isinstance(model.method, BaseMethod)
        assert model.trainer is None

        # After training, trainer should be set
        with patch("sc_flow.trainer._trainer.Trainer"):
            model.train(adata, n_train_steps=1)
        assert model.trainer is not None
