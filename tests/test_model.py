import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from sc_flow import SCFlow
from sc_flow.data._manager import DataManager
from sc_flow.methods._methods import BaseMethod


# -----------------------------------------------------------------------------
# Dummy module (picklable, no recursion) for most tests
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
        return []  # empty list – but we will mock optimizer creation for most tests

    def train(self):
        pass

    def eval(self):
        pass


# -----------------------------------------------------------------------------
# Helper to create an AnnData with a continuous covariate for condition space
# -----------------------------------------------------------------------------
def _add_continuous_covariate(adata: AnnData, key: str = "X_repr", n_dim: int = 10) -> AnnData:
    """Add a random continuous covariate to adata.obsm."""
    adata.obsm[key] = np.random.randn(adata.n_obs, n_dim)
    return adata


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
    def __init__(self, dims_registry, dm, is_paired_setting, *args, **kwargs):
        self._dims_registry = dims_registry
        self._dm = dm
        self._is_paired_setting = is_paired_setting
        self._module = DummyModule()
        self._train_mode = True

    def extract_state_data(self, matched_distr):
        """Return dummy StateData from the target state of the matched distribution."""
        from sc_flow.data.containers._state import StateData

        # Dummy state: zeros with correct number of observations and feature dimension
        n_obs = len(matched_distr.target_distr.ann_df)
        n_feat = len(self._dims_registry.feature_names)
        X = np.zeros((n_obs, n_feat))
        return StateData(X)

    def set_train_mode(self, mode: bool):
        self._train_mode = mode

    def train_step(self, *args, **kwargs):
        import torch

        return torch.tensor(0.0), {"loss": 0.0}

    def predict(self, matched_distr, *args, **kwargs):
        n_obs = len(matched_distr.target_distr.ann_df)
        n_feat = len(self._dims_registry.feature_names)
        samples = np.zeros((n_obs, n_feat))
        return DummyPredictionData(samples)


# -----------------------------------------------------------------------------
# Fixture to mock the optimization manager creation (only for tests that need it)
# -----------------------------------------------------------------------------
@pytest.fixture
def mock_optim_manager():
    """Prevent real optimizer creation in tests that don't need real training."""
    with patch("sc_flow.backends.torch.methods._opt.TorchOptimizationManager.from_config") as mock:
        mock_manager = MagicMock()
        mock_manager.step = MagicMock()
        mock.return_value = mock_manager
        yield mock_manager


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
        SCFlow.register_adata(adata, control_values_dict={"drugA": "control"})
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
        mock_registry = {"cfm": DummyMethod}
        monkeypatch.setattr("sc_flow.backends.torch.methods.METHODS_REGISTRY", mock_registry)
        SCFlow.register_adata(adata)
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
    def test_train_calls_trainer_and_sets_mode(self, adata: AnnData, mock_optim_manager):
        SCFlow.register_adata(adata)
        model = SCFlow(method_cls=DummyMethod, backend="torch")
        with patch("sc_flow._model.Trainer") as mock_trainer_cls:
            mock_trainer = mock_trainer_cls.return_value
            set_train_mode_spy = MagicMock(wraps=model.method.set_train_mode)
            model.method.set_train_mode = set_train_mode_spy
            model.train(adata, n_train_steps=10, valid_freq=5, train_batch_size=32, sort=True)

            mock_trainer_cls.assert_called_once()
            args, _ = mock_trainer_cls.call_args
            assert args[0] is model.method
            assert args[1] is mock_optim_manager
            set_train_mode_spy.assert_called_once_with(True)
            mock_trainer.train.assert_called_once()
            call_kwargs = mock_trainer.train.call_args[1]
            assert call_kwargs["n_train_steps"] == 10
            assert call_kwargs["valid_freq"] == 5

    def test_train_with_validation_samplers(self, adata: AnnData, mock_optim_manager):
        SCFlow.register_adata(adata)
        model = SCFlow(method_cls=DummyMethod)
        with patch("sc_flow._model.Trainer") as mock_trainer_cls:
            mock_trainer = mock_trainer_cls.return_value
            val_adatas = {"val1": adata, "val2": adata}
            model.train(adata, val_adatas_dict=val_adatas, n_train_steps=5, sort=True)

            args, _ = mock_trainer.train.call_args
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
        pred_adata = model.predict(adata, sort=True)
        assert isinstance(pred_adata, AnnData)
        assert pred_adata.n_obs == adata.n_obs
        assert pred_adata.n_vars == adata.n_vars
        pd.testing.assert_index_equal(pred_adata.obs.index, adata.obs.index)

    def test_predict_with_return_tensors(self, adata: AnnData):
        SCFlow.register_adata(adata)
        model = SCFlow(method_cls=DummyMethod)
        result = model.predict(adata, return_tensors=True, sort=True)
        assert isinstance(result, tuple) and len(result) == 2
        pred_adata, pred_data = result
        assert isinstance(pred_adata, AnnData)
        assert hasattr(pred_data, "samples") and hasattr(pred_data, "traj")

    def test_predict_empty_tree_returns_empty_anndata(self, adata: AnnData, monkeypatch):
        SCFlow.register_adata(adata)
        model = SCFlow(method_cls=DummyMethod)
        mock_tree = MagicMock()
        mock_tree.flatten.return_value = ()
        # The lambda must accept any keyword arguments (sort, view_on_condition_space, ...)
        monkeypatch.setattr(model._dm, "compile_adata", lambda x, **kwargs: mock_tree)
        pred_adata = model.predict(adata, sort=True)
        assert pred_adata.n_obs == 0
        assert pred_adata.n_vars == adata.n_vars

    def test_predict_sets_eval_mode(self, adata: AnnData):
        SCFlow.register_adata(adata)
        model = SCFlow(method_cls=DummyMethod)
        set_train_mode_spy = MagicMock(wraps=model.method.set_train_mode)
        model.method.set_train_mode = set_train_mode_spy
        model.predict(adata, sort=True)
        set_train_mode_spy.assert_called_once_with(False)

    # --------------------------------------------------------------------------
    # Properties
    # --------------------------------------------------------------------------
    def test_properties(self, adata: AnnData, mock_optim_manager):
        SCFlow.register_adata(adata)
        model = SCFlow(method_cls=DummyMethod)
        assert isinstance(model.dm, DataManager)
        assert model.is_paired_setting is False
        assert isinstance(model.method, BaseMethod)
        assert model.trainer is None
        with patch("sc_flow._model.Trainer") as _mock_trainer_cls:
            model.train(adata, n_train_steps=1, sort=True)
        assert model.trainer is not None

    # --------------------------------------------------------------------------
    # Save / Load (without mocking the optimizer manager)
    # --------------------------------------------------------------------------
    def test_save_load_and_predict(self, adata):
        SCFlow.register_adata(adata)
        model = SCFlow(method_cls=DummyMethod, backend="torch")
        pred1 = model.predict(adata, sort=True)

        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
            tmp_path = tmp.name
        model.save(tmp_path, allow_overwrite=True)

        loaded = SCFlow.load(tmp_path, map_location="cpu")
        pred2 = loaded.predict(adata, sort=True)

        np.testing.assert_array_equal(pred1.X, pred2.X)
        os.unlink(tmp_path)

    def test_save_load_and_continue_training(self, adata):
        """Test save/load with a real tiny module (not a mock)."""
        torch = pytest.importorskip("torch")

        # Define a real torch module with parameters
        class RealDummyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)

            def forward(self, t, x, condition_dict=None, source=None):
                return self.linear(x)

            @classmethod
            def init_from_dims_registry(cls, dims_registry, *args, **kwargs):
                return cls()

        # Create a proper method class that uses RealDummyModule
        class RealDummyMethod(BaseMethod):
            def build_module(self, *args, **kwargs):
                return RealDummyModule.init_from_dims_registry(self._dims_registry, *args, **kwargs)

            def __init__(self, dims_registry, dm, is_paired_setting, *args, **kwargs):
                super().__init__(dims_registry, dm, is_paired_setting, *args, **kwargs)

            def extract_state_data(self, matched_distr):
                from sc_flow.data.containers._state import StateData

                n_obs = len(matched_distr.target_distr.ann_df)
                n_feat = len(self._dims_registry.feature_names)
                X = np.zeros((n_obs, n_feat))
                return StateData(X)

            def set_train_mode(self, mode: bool):
                if mode:
                    self._module.train()
                else:
                    self._module.eval()

            def train_step(self, matched_distr, *args, **kwargs):
                # Create a dummy input that requires grad
                dummy_x = torch.randn(4, 2, requires_grad=True)
                t = torch.tensor(0.5, requires_grad=False)
                output = self._module(t, dummy_x)
                loss = output.sum()
                return loss, {"loss": loss.item()}

            def predict(self, matched_distr, *args, **kwargs):
                n_obs = len(matched_distr.target_distr.ann_df)
                n_feat = len(self._dims_registry.feature_names)
                samples = np.zeros((n_obs, n_feat))
                from tests.test_model import DummyPredictionData

                return DummyPredictionData(samples)

        SCFlow.register_adata(adata)
        model = SCFlow(method_cls=RealDummyMethod, backend="torch")
        model.train(adata, n_train_steps=5, train_batch_size=4, sort=True)

        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
            tmp_path = tmp.name
        model.save(tmp_path, allow_overwrite=True)

        loaded = SCFlow.load(tmp_path, map_location="cpu")
        loaded.train(adata, n_train_steps=5, train_batch_size=4)

        os.unlink(tmp_path)


class TestSCFlowConditionSpace:
    """Test SCFlow with view_on_condition_space=True using adata."""

    def test_register_adata_with_condition_space(self, adata: AnnData):
        """Registering with condition space should set class attributes."""
        adata = _add_continuous_covariate(adata)
        SCFlow.register_adata(
            adata,
            view_on_condition_space=True,
            condition_state_key="X_repr",
            conditions={"drug": ("drugA",)},
            conditions_reps={"drug": "drug"},
            conditions_covariates=["X_repr"],
            groups=("source_split",),
            groups_reps={"source_split": "source_split"},
        )
        assert SCFlow._view_on_condition_space_cls is True
        assert SCFlow._condition_state_key_cls == "X_repr"
        # Clean up class attributes for subsequent tests
        SCFlow._dm_cls = None
        SCFlow._dims_registry = None
        SCFlow._is_paired_setting_cls = False
        SCFlow._view_on_condition_space_cls = False
        SCFlow._condition_state_key_cls = None

    def test_train_with_condition_space(self, adata: AnnData, mock_optim_manager):
        """Training with condition space should correctly transform the data."""
        adata = _add_continuous_covariate(adata)
        SCFlow.register_adata(
            adata,
            view_on_condition_space=True,
            condition_state_key="X_repr",
            conditions={"drug": ("drugA",)},
            conditions_reps={"drug": "drug"},
            conditions_covariates=["X_repr"],
            groups=("source_split",),
            groups_reps={"source_split": "source_split"},
        )
        model = SCFlow(method_cls=DummyMethod, backend="torch")

        # Create a spy on compile_adata to record calls while preserving original behavior
        original_compile = model._dm.compile_adata
        mock_compile = MagicMock(wraps=original_compile)
        model._dm.compile_adata = mock_compile

        with patch("sc_flow._model.Trainer") as mock_trainer_cls:
            mock_trainer = mock_trainer_cls.return_value
            model.train(adata, n_train_steps=10, train_batch_size=32, sort=True)

            # Verify compile_adata was called with the correct flags
            mock_compile.assert_called_once()
            _, called_kwargs = mock_compile.call_args
            assert called_kwargs["view_on_condition_space"] is True
            assert called_kwargs["condition_state_key"] == "X_repr"

            # Also ensure training was invoked (optional, but good)
            mock_trainer.train.assert_called_once()

        # Clean up class attributes for subsequent tests
        SCFlow._dm_cls = None
        SCFlow._dims_registry = None
        SCFlow._is_paired_setting_cls = False
        SCFlow._view_on_condition_space_cls = False
        SCFlow._condition_state_key_cls = None

    def test_predict_with_condition_space(self, adata: AnnData):
        """Predict with condition space should yield predictions with correct shape."""
        adata = _add_continuous_covariate(adata)
        SCFlow.register_adata(
            adata,
            view_on_condition_space=True,
            condition_state_key="X_repr",
            conditions={"drug": ("drugA",)},
            conditions_reps={"drug": "drug"},
            conditions_covariates=["X_repr"],
            groups=("source_split",),
            groups_reps={"source_split": "source_split"},
        )
        model = SCFlow(method_cls=DummyMethod, backend="torch")
        pred_adata = model.predict(adata, sort=True)

        assert isinstance(pred_adata, AnnData)
        assert pred_adata.n_obs == adata.n_obs
        # The feature dimension should now be the continuous covariate's dimension
        assert pred_adata.n_vars == adata.obsm["X_repr"].shape[1]

        # Clean up
        SCFlow._dm_cls = None
        SCFlow._dims_registry = None
        SCFlow._is_paired_setting_cls = False
        SCFlow._view_on_condition_space_cls = False
        SCFlow._condition_state_key_cls = None

    def test_save_load_with_condition_space(self, adata: AnnData):
        """Save and load model that was configured with condition space."""
        adata = _add_continuous_covariate(adata)
        SCFlow.register_adata(
            adata,
            view_on_condition_space=True,
            condition_state_key="X_repr",
            conditions={"drug": ("drugA",)},
            conditions_reps={"drug": "drug"},
            conditions_covariates=["X_repr"],
            groups=("source_split",),
            groups_reps={"source_split": "source_split"},
        )
        model = SCFlow(method_cls=DummyMethod, backend="torch")
        pred1 = model.predict(adata, sort=True)

        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
            tmp_path = tmp.name
        model.save(tmp_path, allow_overwrite=True)

        # Load with the same adata (must re-register)
        loaded = SCFlow.load(
            tmp_path,
            adata=adata,
            view_on_condition_space=True,
            condition_state_key="X_repr",
            conditions={"drug": ("drugA",)},
            conditions_reps={"drug": "drug"},
            conditions_covariates=["X_repr"],
            groups=("source_split",),
            groups_reps={"source_split": "source_split"},
        )
        pred2 = loaded.predict(adata)

        np.testing.assert_array_equal(pred1.X, pred2.X)
        os.unlink(tmp_path)

        # Clean up
        SCFlow._dm_cls = None
        SCFlow._dims_registry = None
        SCFlow._is_paired_setting_cls = False
        SCFlow._view_on_condition_space_cls = False
        SCFlow._condition_state_key_cls = None


class TestSCFlowPredictCombinations:
    @pytest.fixture(autouse=True)
    def _reset_class_state(self):
        """Ensure SCFlow class attributes are clean before/after each test."""
        SCFlow._dm_cls = None
        SCFlow._dims_registry = None
        SCFlow._is_paired_setting_cls = False
        SCFlow._view_on_condition_space_cls = False
        SCFlow._condition_state_key_cls = None
        yield
        SCFlow._dm_cls = None
        SCFlow._dims_registry = None
        SCFlow._is_paired_setting_cls = False
        SCFlow._view_on_condition_space_cls = False
        SCFlow._condition_state_key_cls = None

    @pytest.mark.parametrize(
        "has_cont_cond, has_cat_cond, has_groups, has_source, view_on_condition_space",
        [
            (False, False, False, False, False),
            (False, False, False, True, False),
            (False, False, True, False, False),
            (False, False, True, True, False),
            (False, True, False, False, False),
            (False, True, False, True, False),
            (False, True, True, False, False),
            (False, True, True, True, False),
            (True, False, False, False, False),
            (True, False, False, False, True),
            (True, False, False, True, False),
            (True, False, False, True, True),
            (True, True, False, False, False),
            (True, True, False, False, True),
            (True, True, False, True, False),
            (True, True, False, True, True),
            (True, False, True, False, False),
            (True, False, True, False, True),
            (True, False, True, True, False),
            (True, False, True, True, True),
            (True, True, True, False, False),
            (True, True, True, False, True),
            (True, True, True, True, False),
            (True, True, True, True, True),
        ],
    )
    def test_predict_combinations(
        self,
        adata,
        has_cont_cond,
        has_cat_cond,
        has_groups,
        has_source,
        view_on_condition_space,
    ):
        """Test prediction with all schema feature combinations."""
        # Skip invalid: view_on_condition_space requires has_cont_cond
        if view_on_condition_space and not has_cont_cond:
            pytest.skip("view_on_condition_space requires a continuous condition covariate")

        adata = adata.copy()
        base_n_obs = adata.n_obs

        # Keep only columns that we actually need
        keep_cols = []
        if has_cat_cond:
            keep_cols.append("drugA")
        if has_groups:
            keep_cols.append("source_split")
        if keep_cols:
            adata.obs = adata.obs[keep_cols]
        else:
            adata.obs = pd.DataFrame(index=adata.obs_names)

        # Build condition and group specifications
        conditions = {}
        conditions_reps = {}
        conditions_covariates = [] if has_cont_cond else None
        groups = None
        groups_reps = {}
        control_values_dict = None

        # Add categorical condition column if requested
        if has_cat_cond:
            realm_col = "drug"
            cat_col = "drugA"
            control_val = "control"
            unique_vals = adata.obs[cat_col].unique()
            rep_dim = 4
            adata.uns[realm_col] = {val: np.random.randn(rep_dim) for val in unique_vals}
            if has_source:
                if control_val not in adata.uns[realm_col]:
                    adata.uns[realm_col][control_val] = np.random.randn(rep_dim)
                control_values_dict = {realm_col: control_val}
            conditions[realm_col] = (cat_col,)
            conditions_reps[realm_col] = realm_col

        # Add continuous condition if requested
        cont_key = "X_repr"
        if has_cont_cond:
            adata = _add_continuous_covariate(adata, key=cont_key, n_dim=5)
            conditions_covariates = [cont_key]

        # Add groups if requested
        if has_groups:
            group_col = "source_split"
            groups = (group_col,)
            groups_reps[group_col] = group_col
            unique_groups = adata.obs[group_col].unique()
            adata.uns[group_col] = {val: np.random.randn(2) for val in unique_groups}

        # For paired setting without a categorical condition, create a dummy column
        if has_source and not has_cat_cond:
            dummy_col = "dummy_paired"
            n_obs = len(adata)
            n_control = n_obs // 2
            control_vals = ["control"] * n_control
            treatment_vals = ["treatment"] * (n_obs - n_control)
            adata.obs[dummy_col] = control_vals + treatment_vals
            adata.obs[dummy_col] = adata.obs[dummy_col].astype("category")
            conditions[dummy_col] = (dummy_col,)
            conditions_reps[dummy_col] = dummy_col
            adata.uns[dummy_col] = {"control": np.random.randn(2), "treatment": np.random.randn(2)}
            control_values_dict = {dummy_col: "control"}

        # Register the data
        SCFlow.register_adata(
            adata,
            view_on_condition_space=view_on_condition_space,
            condition_state_key=cont_key if view_on_condition_space else None,
            conditions=conditions,
            conditions_reps=conditions_reps,
            conditions_covariates=conditions_covariates,
            groups=groups,
            groups_reps=groups_reps,
            control_values_dict=control_values_dict,
        )

        model = SCFlow(method_cls=DummyMethod, backend="torch")

        # Capture the matched distribution
        captured_matched_distr = []
        original_predict = model.method.predict

        def spy_predict(matched_distr, *args, **kwargs):
            captured_matched_distr.append(matched_distr)
            return original_predict(matched_distr, *args, **kwargs)

        model.method.predict = spy_predict

        pred_adata = model.predict(adata, sort=True)

        # --- Compute expected number of observations ---
        if has_source:
            # For paired setting: only treatment cells (non-control) are predicted.
            control_dict = model._dm.control_values_dict
            cond_schema = model._dm.condition_data_schema
            realm = next(iter(control_dict.keys()))
            control_val = control_dict[realm]
            cols = cond_schema.conditions[realm]  # tuple of column names
            col = cols[0]  # our tests use a single column
            expected_n_obs = (adata.obs[col] != control_val).sum()
        else:
            # No pairing: all cells are predicted.
            expected_n_obs = base_n_obs

        # --- Assertions ---
        # 1. Output AnnData has correct number of observations
        assert pred_adata.n_obs == expected_n_obs

        # 2. Feature dimension
        if view_on_condition_space:
            expected_n_vars = adata.obsm[cont_key].shape[1]
        else:
            expected_n_vars = len(SCFlow._dims_registry.feature_names)
        assert pred_adata.n_vars == expected_n_vars

        # 3. If view_on_condition_space, verify condition_data contents
        if view_on_condition_space:
            matched = captured_matched_distr[0]
            cond_data = matched.target_distr.condition_data

            # The state key must have been removed from continuous covariates
            if cond_data is not None and cond_data.continuous_covariates is not None:
                assert cont_key not in cond_data.continuous_covariates.mapping

            # Additional checks depending on presence of other conditions
            # We don't enforce emptiness because a dummy categorical condition may exist.
            # Groups must be as requested
            if has_groups:
                assert matched.target_distr.groups_data is not None
                assert group_col in matched.target_distr.groups_data.ann_df.columns
            else:
                groups_obj = matched.target_distr.groups_data
                # Accept None or a groups object with zero columns
                if groups_obj is not None:
                    assert len(groups_obj.ann_df.columns) == 0

        # 4. Source distribution presence
        matched = captured_matched_distr[0]
        if has_source:
            assert matched.source is not None
        else:
            assert matched.source is None

    def test_condition_space_preserves_all_other_covariates(self, adata):
        """Explicit test that after view_on_condition_space, all non-state condition
        covariates (categorical and continuous) and groups remain."""
        adata = adata.copy()
        # Add two continuous covariates: one will be the state, one a regular condition
        cond_state_key = "X_paired_condition"
        cond_key = "paired_condition"
        adata.obsm[cond_state_key] = np.random.randn(adata.n_obs, 4)
        adata.obsm[cond_key] = np.random.randn(adata.n_obs, 3)

        # Use categorical condition and groups
        cat_cond_col = "drugA"
        group_col = "source_split"

        # Keep only the relevant columns in obs to avoid automatic detection
        adata.obs = adata.obs[[cat_cond_col, group_col]].copy()

        # Add dummy representations in uns for categorical condition and groups
        unique_cat = adata.obs[cat_cond_col].unique()
        adata.uns[cat_cond_col] = {val: np.random.randn(2) for val in unique_cat}
        unique_group = adata.obs[group_col].unique()
        adata.uns[group_col] = {val: np.random.randn(2) for val in unique_group}

        conditions = {cat_cond_col: (cat_cond_col,)}
        conditions_reps = {cat_cond_col: cat_cond_col}
        conditions_covariates = [cond_state_key, cond_key]
        groups = (group_col,)
        groups_reps = {group_col: group_col}

        SCFlow.register_adata(
            adata,
            view_on_condition_space=True,
            condition_state_key=cond_state_key,
            conditions=conditions,
            conditions_reps=conditions_reps,
            conditions_covariates=conditions_covariates,
            groups=groups,
            groups_reps=groups_reps,
        )

        model = SCFlow(method_cls=DummyMethod, backend="torch")

        # Use DataManager.sort_adata to get the sorted version of adata
        # This matches the internal sorting when sort=True is passed to predict.
        sorted_adata = model._dm.sort_adata(adata)
        sorted_cond_array = sorted_adata.obsm[cond_key]

        captured = []
        original_predict = model.method.predict

        def spy(matched_distr, *args, **kwargs):
            captured.append(matched_distr)
            return original_predict(matched_distr, *args, **kwargs)

        model.method.predict = spy

        model.predict(adata, sort=True)

        # Collect continuous condition arrays from all nodes and concatenate
        cond_arrays = []
        for match in captured:
            cond_data = match.target_distr.condition_data
            assert cond_data is not None
            assert cond_data.continuous_covariates is not None
            assert cond_key in cond_data.continuous_covariates.mapping
            assert cond_state_key not in cond_data.continuous_covariates.mapping
            cond_arrays.append(cond_data.continuous_covariates.mapping[cond_key])

        concatenated_cond_array = np.vstack(cond_arrays)
        np.testing.assert_array_equal(concatenated_cond_array, sorted_cond_array)

        # Also verify each node has categorical and groups data
        for match in captured:
            cond_data = match.target_distr.condition_data
            assert cond_data.categorical_covariates is not None
            assert cat_cond_col in cond_data.categorical_covariates.ann_df.columns

            assert match.target_distr.groups_data is not None
            assert group_col in match.target_distr.groups_data.ann_df.columns


class TestSCFlowPredictMatchedKeys:
    """Test the matched_keys argument in SCFlow.predict."""

    @pytest.fixture(autouse=True)
    def _reset_class_state(self):
        """Ensure SCFlow class attributes are clean before/after each test."""
        SCFlow._dm_cls = None
        SCFlow._dims_registry = None
        SCFlow._is_paired_setting_cls = False
        SCFlow._view_on_condition_space_cls = False
        SCFlow._condition_state_key_cls = None
        yield
        SCFlow._dm_cls = None
        SCFlow._dims_registry = None
        SCFlow._is_paired_setting_cls = False
        SCFlow._view_on_condition_space_cls = False
        SCFlow._condition_state_key_cls = None

    def _setup_paired_data(self, adata, has_continuous=False):
        """Create a paired dataset with drug condition and cell_line groups."""
        adata = adata.copy()
        # Use existing drugA column for conditions
        adata.obs = adata.obs[["drugA", "source_split"]].copy()
        # Ensure 'control' values exist
        adata.obs["drugA"] = adata.obs["drugA"].astype(str)
        # Set half of the rows to 'control'
        n_obs = len(adata)
        control_vals = ["control"] * (n_obs // 2)
        treatment_vals = ["treatment"] * (n_obs - n_obs // 2)
        adata.obs["drugA"] = control_vals + treatment_vals
        # Add representation for the categorical condition
        adata.uns["drug"] = {"control": np.random.randn(4), "treatment": np.random.randn(4)}
        # Groups representation
        unique_groups = adata.obs["source_split"].unique()
        adata.uns["source_split"] = {g: np.random.randn(2) for g in unique_groups}
        if has_continuous:
            adata.obsm["X_repr"] = np.random.randn(n_obs, 5)
        return adata

    def test_predict_with_matched_keys_override(self, adata):
        """Passing matched_keys to predict overrides the instance's matched_keys."""
        adata = self._setup_paired_data(adata)
        # Register with instance matched_keys
        instance_keys = {("control",): ("treatment",)}
        SCFlow.register_adata(
            adata,
            conditions={"drug": ("drugA",)},
            conditions_reps={"drug": "drug"},
            groups=("source_split",),
            groups_reps={"source_split": "source_split"},
            control_values_dict={"drug": "control"},
            matched_keys=instance_keys,
        )
        model = SCFlow(method_cls=DummyMethod, backend="torch")
        # Override with different keys at predict time
        override_keys = {("control",): ("treatment",)}  # same for simplicity; could be different
        pred_adata = model.predict(adata, sort=True, matched_keys=override_keys)
        # Check that only treatment rows are predicted (since pairing uses override)
        # In this simple setup, target condition should be 'treatment'
        assert pred_adata.n_obs > 0
        assert all(pred_adata.obs["drugA"] == "treatment")

    def test_predict_matched_keys_none_uses_instance(self, adata):
        """When matched_keys=None, the instance's matched_keys is used."""
        adata = self._setup_paired_data(adata)
        instance_keys = {("control",): ("treatment",)}
        SCFlow.register_adata(
            adata,
            conditions={"drug": ("drugA",)},
            conditions_reps={"drug": "drug"},
            groups=("source_split",),
            groups_reps={"source_split": "source_split"},
            control_values_dict={"drug": "control"},
            matched_keys=instance_keys,
        )
        model = SCFlow(method_cls=DummyMethod, backend="torch")
        pred_adata = model.predict(adata, sort=True, matched_keys=None)
        assert pred_adata.n_obs > 0
        assert all(pred_adata.obs["drugA"] == "treatment")

    def test_predict_matched_keys_without_control_values(self, adata):
        """When control_values_dict is None, matched_keys can still define source-target pairs."""
        adata = self._setup_paired_data(adata)
        # No control_values_dict, only matched_keys
        keys = {("control",): ("treatment",)}
        SCFlow.register_adata(
            adata,
            conditions={"drug": ("drugA",)},
            conditions_reps={"drug": "drug"},
            groups=("source_split",),
            groups_reps={"source_split": "source_split"},
            matched_keys=keys,  # no control_values_dict
        )
        model = SCFlow(method_cls=DummyMethod, backend="torch")
        pred_adata = model.predict(adata, sort=True, matched_keys=keys)
        assert pred_adata.n_obs > 0
        assert all(pred_adata.obs["drugA"] == "treatment")

    def test_predict_matched_keys_on_condition_view_paired_enabled(self, adata):
        """With view_on_condition_space=True and allow_paired_settings_on_condition_view=True,
        matched_keys are honored."""
        adata = self._setup_paired_data(adata, has_continuous=True)
        keys = {("control",): ("treatment",)}
        SCFlow.register_adata(
            adata,
            view_on_condition_space=True,
            condition_state_key="X_repr",
            conditions={"drug": ("drugA",)},
            conditions_reps={"drug": "drug"},
            conditions_covariates=["X_repr"],
            groups=("source_split",),
            groups_reps={"source_split": "source_split"},
            control_values_dict={"drug": "control"},
            matched_keys=keys,
            allow_paired_settings_on_condition_view=True,
        )
        model = SCFlow(method_cls=DummyMethod, backend="torch")
        pred_adata = model.predict(
            adata,
            sort=True,
            view_on_condition_space=True,
            condition_state_key="X_repr",
            matched_keys=keys,
        )
        assert pred_adata.n_obs > 0
        assert all(pred_adata.obs["drugA"] == "treatment")

    def test_predict_matched_keys_on_condition_view_paired_disabled(self, adata):
        """When view_on_condition_space=True and allow_paired_settings_on_condition_view=False,
        matched_keys are ignored → no source, full dataset predicted."""
        adata = self._setup_paired_data(adata, has_continuous=True)
        keys = {("control",): ("treatment",)}
        SCFlow.register_adata(
            adata,
            view_on_condition_space=True,
            condition_state_key="X_repr",
            conditions={"drug": ("drugA",)},
            conditions_reps={"drug": "drug"},
            conditions_covariates=["X_repr"],
            groups=("source_split",),
            groups_reps={"source_split": "source_split"},
            control_values_dict={"drug": "control"},
            matched_keys=keys,
            allow_paired_settings_on_condition_view=False,  # disabled
        )
        model = SCFlow(method_cls=DummyMethod, backend="torch")
        pred_adata = model.predict(
            adata,
            sort=True,
            view_on_condition_space=True,
            condition_state_key="X_repr",
            matched_keys=keys,
        )
        # With pairing disabled, no source → all cells predicted (both control and treatment)
        # So we should see both conditions in output.
        assert pred_adata.n_obs == len(adata)
        assert set(pred_adata.obs["drugA"].unique()) == {"control", "treatment"}

    def test_predict_matched_keys_invalid_target_raises(self, adata):
        """If matched_keys contains a target key not present in the data, an error should be raised."""
        adata = self._setup_paired_data(adata)
        keys = {("control",): ("nonexistent",)}
        SCFlow.register_adata(
            adata,
            conditions={"drug": ("drugA",)},
            conditions_reps={"drug": "drug"},
            groups=("source_split",),
            groups_reps={"source_split": "source_split"},
            control_values_dict={"drug": "control"},
        )
        model = SCFlow(method_cls=DummyMethod, backend="torch")
        with pytest.raises(KeyError, match="nonexistent"):
            model.predict(adata, sort=True, matched_keys=keys)


class TestSCFlowPredictControlValues:
    """Test the control_values_dict argument in SCFlow.predict."""

    @pytest.fixture(autouse=True)
    def _reset_class_state(self):
        """Ensure SCFlow class attributes are clean before/after each test."""
        SCFlow._dm_cls = None
        SCFlow._dims_registry = None
        SCFlow._is_paired_setting_cls = False
        SCFlow._view_on_condition_space_cls = False
        SCFlow._condition_state_key_cls = None
        yield
        SCFlow._dm_cls = None
        SCFlow._dims_registry = None
        SCFlow._is_paired_setting_cls = False
        SCFlow._view_on_condition_space_cls = False
        SCFlow._condition_state_key_cls = None

    def _setup_paired_data(self, adata, has_continuous=False):
        """Create a paired dataset with drug condition and source_split groups."""
        adata = adata.copy()
        adata.obs = adata.obs[["drugA", "source_split"]].copy()
        adata.obs["drugA"] = adata.obs["drugA"].astype(str)
        n_obs = len(adata)
        control_vals = ["control"] * (n_obs // 2)
        treatment_vals = ["treatment"] * (n_obs - n_obs // 2)
        adata.obs["drugA"] = control_vals + treatment_vals
        adata.uns["drug"] = {"control": np.random.randn(4), "treatment": np.random.randn(4)}
        unique_groups = adata.obs["source_split"].unique()
        adata.uns["source_split"] = {g: np.random.randn(2) for g in unique_groups}
        if has_continuous:
            adata.obsm["X_repr"] = np.random.randn(n_obs, 5)
        return adata

    def test_predict_control_values_override(self, adata):
        """Passing control_values_dict to predict overrides the instance's control_values_dict."""
        adata = self._setup_paired_data(adata)
        SCFlow.register_adata(
            adata,
            conditions={"drug": ("drugA",)},
            conditions_reps={"drug": "drug"},
            groups=("source_split",),
            groups_reps={"source_split": "source_split"},
            control_values_dict={"drug": "control"},
        )
        model = SCFlow(method_cls=DummyMethod, backend="torch")
        override_dict = {"drug": "control"}  # same value; test that override is used
        pred_adata = model.predict(adata, sort=True, control_values_dict=override_dict)
        assert pred_adata.n_obs > 0
        assert all(pred_adata.obs["drugA"] == "treatment")

    def test_predict_control_values_none_uses_instance(self, adata):
        """When control_values_dict=None, the instance's control_values_dict is used."""
        adata = self._setup_paired_data(adata)
        SCFlow.register_adata(
            adata,
            conditions={"drug": ("drugA",)},
            conditions_reps={"drug": "drug"},
            groups=("source_split",),
            groups_reps={"source_split": "source_split"},
            control_values_dict={"drug": "control"},
        )
        model = SCFlow(method_cls=DummyMethod, backend="torch")
        pred_adata = model.predict(adata, sort=True, control_values_dict=None)
        assert pred_adata.n_obs > 0
        assert all(pred_adata.obs["drugA"] == "treatment")

    def test_predict_control_values_without_instance(self, adata):
        """When instance has no control_values_dict, a custom dict works."""
        adata = self._setup_paired_data(adata)
        SCFlow.register_adata(
            adata,
            conditions={"drug": ("drugA",)},
            conditions_reps={"drug": "drug"},
            groups=("source_split",),
            groups_reps={"source_split": "source_split"},
            # no control_values_dict
        )
        model = SCFlow(method_cls=DummyMethod, backend="torch")
        custom_dict = {"drug": "control"}
        pred_adata = model.predict(adata, sort=True, control_values_dict=custom_dict)
        assert pred_adata.n_obs > 0
        assert all(pred_adata.obs["drugA"] == "treatment")

    def test_predict_control_values_on_condition_view_paired_enabled(self, adata):
        """With view_on_condition_space=True and allow_paired_settings_on_condition_view=True,
        control_values_dict is honored."""
        adata = self._setup_paired_data(adata, has_continuous=True)
        SCFlow.register_adata(
            adata,
            view_on_condition_space=True,
            condition_state_key="X_repr",
            conditions={"drug": ("drugA",)},
            conditions_reps={"drug": "drug"},
            conditions_covariates=["X_repr"],
            groups=("source_split",),
            groups_reps={"source_split": "source_split"},
            control_values_dict={"drug": "control"},
            allow_paired_settings_on_condition_view=True,
        )
        model = SCFlow(method_cls=DummyMethod, backend="torch")
        pred_adata = model.predict(
            adata,
            sort=True,
            view_on_condition_space=True,
            condition_state_key="X_repr",
            control_values_dict={"drug": "control"},
        )
        assert pred_adata.n_obs > 0
        assert all(pred_adata.obs["drugA"] == "treatment")

    def test_predict_control_values_on_condition_view_paired_disabled(self, adata):
        """When view_on_condition_space=True and allow_paired_settings_on_condition_view=False,
        control_values_dict is ignored → all cells predicted."""
        adata = self._setup_paired_data(adata, has_continuous=True)
        SCFlow.register_adata(
            adata,
            view_on_condition_space=True,
            condition_state_key="X_repr",
            conditions={"drug": ("drugA",)},
            conditions_reps={"drug": "drug"},
            conditions_covariates=["X_repr"],
            groups=("source_split",),
            groups_reps={"source_split": "source_split"},
            control_values_dict={"drug": "control"},
            allow_paired_settings_on_condition_view=False,
        )
        model = SCFlow(method_cls=DummyMethod, backend="torch")
        pred_adata = model.predict(
            adata,
            sort=True,
            view_on_condition_space=True,
            condition_state_key="X_repr",
            control_values_dict={"drug": "control"},
        )
        assert pred_adata.n_obs == len(adata)
        assert set(pred_adata.obs["drugA"].unique()) == {"control", "treatment"}

    def test_predict_control_values_invalid_control_raises(self, adata):
        """If control_values_dict contains a value not present in the data, an error should be raised."""
        adata = self._setup_paired_data(adata)
        SCFlow.register_adata(
            adata,
            conditions={"drug": ("drugA",)},
            conditions_reps={"drug": "drug"},
            groups=("source_split",),
            groups_reps={"source_split": "source_split"},
        )
        model = SCFlow(method_cls=DummyMethod, backend="torch")
        with pytest.raises(KeyError, match="nonexistent"):
            model.predict(adata, sort=True, control_values_dict={"drug": "nonexistent"})


class TestSCFlowPreprocessingIntegration:
    """Verify that the preprocessing pipeline is used correctly by SCFlow."""

    @pytest.fixture(autouse=True)
    def _reset_class_state(self):
        """Ensure SCFlow class attributes are clean before/after each test."""
        SCFlow._dm_cls = None
        SCFlow._dims_registry = None
        SCFlow._is_paired_setting_cls = False
        SCFlow._view_on_condition_space_cls = False
        SCFlow._condition_state_key_cls = None
        yield
        SCFlow._dm_cls = None
        SCFlow._dims_registry = None
        SCFlow._is_paired_setting_cls = False
        SCFlow._view_on_condition_space_cls = False
        SCFlow._condition_state_key_cls = None

    def test_register_adata_fits_and_transforms(self, adata):
        """register_adata must call get_data_dimensionalities with fit_preproc=True
        and apply_transformations=True."""
        with patch.object(DataManager, "get_data_dimensionalities", return_value=MagicMock()) as mock_method:
            SCFlow.register_adata(adata)
        mock_method.assert_called_once()
        call_kwargs = mock_method.call_args.kwargs
        assert call_kwargs["fit_preproc"] is True
        assert call_kwargs["apply_transformations"] is True

    def test_train_applies_transformations_without_refitting(self, adata, mock_optim_manager):
        """train() calls compile_adata with apply_transformations=True and never re-fits."""
        SCFlow.register_adata(adata)
        model = SCFlow(method_cls=DummyMethod, backend="torch")

        with (
            patch.object(model._dm, "compile_adata", wraps=model._dm.compile_adata) as compile_spy,
            patch.object(model._dm._state_preproc, "fit") as fit_spy,
        ):
            with patch("sc_flow._model.Trainer"):
                model.train(adata, n_train_steps=5, sort=True)

        # compile_adata called once for train data
        compile_spy.assert_called_once()
        assert compile_spy.call_args.kwargs["apply_transformations"] is True
        # fit_preproc should not be passed (defaults to False)
        assert compile_spy.call_args.kwargs.get("fit_preproc", False) is False
        fit_spy.assert_not_called()

    def test_predict_applies_transformations_without_refitting(self, adata):
        """predict() calls compile_adata with apply_transformations=True and never re-fits."""
        SCFlow.register_adata(adata)
        model = SCFlow(method_cls=DummyMethod)

        with (
            patch.object(model._dm, "compile_adata", wraps=model._dm.compile_adata) as compile_spy,
            patch.object(model._dm._state_preproc, "fit") as fit_spy,
        ):
            model.predict(adata, sort=True)

        compile_spy.assert_called_once()
        assert compile_spy.call_args.kwargs["apply_transformations"] is True
        assert compile_spy.call_args.kwargs.get("fit_preproc", False) is False
        fit_spy.assert_not_called()

    def test_train_with_validation_also_transforms(self, adata, mock_optim_manager):
        """Validation data must also be compiled with apply_transformations=True."""
        SCFlow.register_adata(adata)
        model = SCFlow(method_cls=DummyMethod)
        val_adata = adata.copy()

        with patch.object(model._dm, "compile_adata", wraps=model._dm.compile_adata) as compile_spy:
            with patch("sc_flow._model.Trainer"):
                model.train(adata, val_adatas_dict={"val": val_adata}, n_train_steps=5, sort=True)

        # compile_adata called for train + validation
        assert compile_spy.call_count == 2
        for call in compile_spy.call_args_list:
            assert call.kwargs["apply_transformations"] is True
            assert call.kwargs.get("fit_preproc", False) is False

    def test_save_unloads_preprocessing_context(self, adata):
        """save() must call dm.unload_preproc() to unload external models."""
        SCFlow.register_adata(adata)
        model = SCFlow(method_cls=DummyMethod, backend="torch")

        with patch.object(model.dm, "unload_preproc") as mock_unload, patch("cloudpickle.dump") as _:
            with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                model.save(tmp_path, allow_overwrite=True)
            finally:
                Path(tmp_path).unlink(missing_ok=True)

        mock_unload.assert_called_once()
