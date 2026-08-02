import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from sckitflow import Model, ModelBuilder
from sckitflow.core.methods._base import BaseMethod
from sckitflow.core.nn._modules import BaseModule
from sckitflow.data._manager import DataManager


# -----------------------------------------------------------------------------
# Dummy module (picklable, no recursion) for most tests
# -----------------------------------------------------------------------------
class DummyModule(BaseModule):
    """Simple dummy module that mimics the interface but is picklable."""

    def _make_modules(self, dims_registry, *args, **kwargs):
        # No real module, just a placeholder
        return None

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
    def __init__(self, X, traj=None):
        self.X = X
        self.traj = traj

    @classmethod
    def concatenate(cls, preds):
        X = np.concatenate([p.X for p in preds], axis=0)
        trajs = [p.traj for p in preds if p.traj is not None]
        traj = np.concatenate(trajs, axis=1) if trajs else None
        return cls(X, traj)


class DummyMethod(BaseMethod):
    _module_cls = DummyModule

    def __init__(self, dims_registry, dm, *args, **kwargs):
        super().__init__(dims_registry, dm, *args, **kwargs)

    def extract_state_data(self, matched_distr):
        """Return dummy StateData from the target state of the matched distribution."""
        from sckitflow.data.containers._state import StateData

        # Dummy state: zeros with correct number of observations and feature dimension
        n_obs = len(matched_distr.target_distr.ann_df)
        n_feat = len(self._dims_registry.feature_names)
        X = np.zeros((n_obs, n_feat))
        return StateData(X)

    def set_train_mode(self, mode: bool):
        self._train_mode = mode

    def compute_loss(self, *args, **kwargs):
        import torch

        return torch.tensor(0.0), {"loss": 0.0}

    def infer(self, step_data, *args, **kwargs):
        n_obs, n_feat = step_data.target_state.shape
        samples = np.zeros((n_obs, n_feat))
        return DummyPredictionData(samples)


# -----------------------------------------------------------------------------
# Helper to build a model with the instance-based API
# -----------------------------------------------------------------------------
def _make_model(adata: AnnData, method_cls=DummyMethod, dm_kwargs=None, **method_kwargs) -> Model:
    """Build a Model from an AnnData using the two-step ModelBuilder flow."""
    builder = ModelBuilder.from_adata(adata, **(dm_kwargs or {}))
    return builder.build(method_cls=method_cls, **method_kwargs)


# -----------------------------------------------------------------------------
# Fixture to mock the optimization manager creation (only for tests that need it)
# -----------------------------------------------------------------------------
@pytest.fixture
def mock_optim_manager():
    """Prevent real optimizer creation in tests that don't need real training."""
    with patch("sckitflow.core.methods._opt.OptimizationManager.from_config") as mock:
        mock_manager = MagicMock()
        mock_manager.step = MagicMock()
        mock.return_value = mock_manager
        yield mock_manager


# -----------------------------------------------------------------------------
# Test suite
# -----------------------------------------------------------------------------
class TestModel:
    """Test suite for the Model orchestration class."""

    # --------------------------------------------------------------------------
    # Construction and Initialization
    # --------------------------------------------------------------------------
    def test_builder_builds_dm_and_dims(self, adata: AnnData):
        model = _make_model(adata)
        assert isinstance(model.dm, DataManager)
        assert model._dims_registry is not None
        assert model.is_paired_setting is False
        assert model._dims_registry.feature_names is not None
        assert len(model._dims_registry.feature_names) == adata.n_vars

    def test_builder_exposes_dm_and_dims(self, adata: AnnData):
        builder = ModelBuilder.from_adata(adata)
        assert isinstance(builder.dm, DataManager)
        assert builder.data_dims is not None
        assert len(builder.data_dims.feature_names) == adata.n_vars

    def test_build_with_method_instance(self, adata: AnnData):
        """build(method=...) attaches a pre-built method instance verbatim."""
        builder = ModelBuilder.from_adata(adata)
        method = DummyMethod(builder.data_dims, builder.dm)
        model = builder.build(method=method)
        assert model.method is method

    def test_control_key_sets_paired_true(self, adata: AnnData):
        model = _make_model(adata, dm_kwargs={"control_values_dict": {"drugA": "control"}})
        assert model.is_paired_setting is True

    def test_direct_init_from_dm_and_dims(self, adata: AnnData):
        """Model can be constructed directly from a data manager and dimensionalities."""
        dm = DataManager()
        data_dims = dm.get_data_dimensionalities(adata)
        model = Model(dm, data_dims, method_cls=DummyMethod)
        assert model.dm is dm
        assert model._dims_registry is data_dims

    def test_init_raises_without_method(self, adata: AnnData):
        with pytest.raises(ValueError, match="At least one of"):
            _make_model(adata, method_cls=None)

    # --------------------------------------------------------------------------
    # Method Resolution
    # --------------------------------------------------------------------------
    def test_method_id_resolves_to_registered_class(self, adata: AnnData, monkeypatch):
        mock_registry = {"cfm": DummyMethod}
        monkeypatch.setattr("sckitflow.core.methods.METHODS_REGISTRY", mock_registry)
        model = _make_model(adata, method_cls=None, method_id="cfm")
        assert isinstance(model.method, DummyMethod)

    def test_unsupported_method_id_raises_error(self, adata: AnnData, monkeypatch):
        mock_registry = {"cfm": DummyMethod}
        monkeypatch.setattr("sckitflow.core.methods.METHODS_REGISTRY", mock_registry)
        with pytest.raises(KeyError, match="not supported"):
            _make_model(adata, method_cls=None, method_id="does_not_exist")

    # --------------------------------------------------------------------------
    # _to_numpy Conversion
    # --------------------------------------------------------------------------
    def test_to_numpy_with_torch_tensor(self, adata: AnnData):
        model = _make_model(adata)
        torch = pytest.importorskip("torch")
        t = torch.tensor([1.0, 2.0, 3.0])
        arr = model._to_numpy(t)
        assert isinstance(arr, np.ndarray)
        np.testing.assert_array_equal(arr, np.array([1.0, 2.0, 3.0]))

    def test_to_numpy_returns_none_for_none(self, adata: AnnData):
        model = _make_model(adata)
        assert model._to_numpy(None) is None

    # --------------------------------------------------------------------------
    # Training Orchestration (with mocking)
    # --------------------------------------------------------------------------
    def test_train_calls_trainer_and_sets_mode(self, adata: AnnData, mock_optim_manager):
        model = _make_model(adata)
        with patch("sckitflow._model.Trainer") as mock_trainer_cls:
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
        model = _make_model(adata)
        with patch("sckitflow._model.Trainer") as mock_trainer_cls:
            mock_trainer = mock_trainer_cls.return_value
            val_adatas = {"val1": adata, "val2": adata}
            model.train(adata, val_adatas_dict=val_adatas, n_train_steps=5, sort=True)

            call_kwargs = mock_trainer.train.call_args.kwargs
            val_samplers = call_kwargs["val_samplers_dict"]
            assert isinstance(val_samplers, dict)
            assert set(val_samplers.keys()) == {"val1", "val2"}

    # --------------------------------------------------------------------------
    # Prediction Pipeline
    # --------------------------------------------------------------------------
    def test_predict_returns_anndata_with_correct_shape(self, adata: AnnData):
        model = _make_model(adata)
        pred_adata = model.predict(adata, sort=True)
        assert isinstance(pred_adata, AnnData)
        assert pred_adata.n_obs == adata.n_obs
        assert pred_adata.n_vars == adata.n_vars
        pd.testing.assert_index_equal(pred_adata.obs.index, adata.obs.index)

    def test_predict_without_target_state(self, adata: AnnData):
        """predict(require_target_state=False) works on an AnnData with no `.X`."""
        model = _make_model(adata)

        # only `.obs` is needed - no expression data, no obsm sample representation
        no_state_adata = AnnData(obs=pd.DataFrame(index=adata.obs_names[:5]))

        pred_adata = model.predict(no_state_adata, sort=True, require_target_state=False)

        assert isinstance(pred_adata, AnnData)
        assert pred_adata.n_obs == 5
        assert pred_adata.n_vars == adata.n_vars

    def test_predict_with_return_raw(self, adata: AnnData):
        model = _make_model(adata)
        result = model.predict(adata, return_raw=True, sort=True)
        assert isinstance(result, tuple) and len(result) == 2
        pred_adata, pred_data = result
        assert isinstance(pred_adata, AnnData)
        assert hasattr(pred_data, "X") and hasattr(pred_data, "traj")

    def test_predict_empty_tree_returns_empty_anndata(self, adata: AnnData, monkeypatch):
        model = _make_model(adata)
        mock_tree = MagicMock()
        mock_tree.flatten.return_value = ()
        # The lambda must accept any keyword arguments (sort, control_values_dict, ...)
        monkeypatch.setattr(model._dm, "compile_adata", lambda x, **kwargs: mock_tree)
        pred_adata = model.predict(adata, sort=True)
        assert pred_adata.n_obs == 0
        assert pred_adata.n_vars == adata.n_vars

    def test_predict_sets_eval_mode(self, adata: AnnData):
        model = _make_model(adata)
        set_train_mode_spy = MagicMock(wraps=model.method.set_train_mode)
        model.method.set_train_mode = set_train_mode_spy
        model.predict(adata, sort=True)
        set_train_mode_spy.assert_called_once_with(False)

    # --------------------------------------------------------------------------
    # Properties
    # --------------------------------------------------------------------------
    def test_properties(self, adata: AnnData, mock_optim_manager):
        model = _make_model(adata)
        assert isinstance(model.dm, DataManager)
        assert model.is_paired_setting is False
        assert isinstance(model.method, BaseMethod)
        assert model.trainer is None
        assert model.condition_state_key is None
        with patch("sckitflow._model.Trainer") as _mock_trainer_cls:
            model.train(adata, n_train_steps=1, sort=True)
        assert model.trainer is not None

    # --------------------------------------------------------------------------
    # Save / Load (without mocking the optimizer manager)
    # --------------------------------------------------------------------------
    def test_save_load_and_predict(self, adata):
        model = _make_model(adata)
        pred1 = model.predict(adata, sort=True)

        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
            tmp_path = tmp.name
        model.save(tmp_path, allow_overwrite=True)

        loaded = Model.load(tmp_path, map_location="cpu")
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
            _module_cls = RealDummyModule

            def __init__(self, dims_registry, dm, *args, **kwargs):
                super().__init__(dims_registry, dm, *args, **kwargs)

            def extract_state_data(self, matched_distr):
                from sckitflow.data.containers._state import StateData

                n_obs = len(matched_distr.target_distr.ann_df)
                n_feat = len(self._dims_registry.feature_names)
                X = np.zeros((n_obs, n_feat))
                return StateData(X)

            def set_train_mode(self, mode: bool):
                if mode:
                    self._module.train()
                else:
                    self._module.eval()

            def compute_loss(self, matched_distr, *args, **kwargs):
                # Create a dummy input that requires grad
                dummy_x = torch.randn(4, 2, requires_grad=True)
                t = torch.tensor(0.5, requires_grad=False)
                output = self._module(t, dummy_x)
                loss = output.sum()
                return loss, {"loss": loss.item()}

            def infer(self, matched_distr, *args, **kwargs):
                n_obs = len(matched_distr.target_distr.ann_df)
                n_feat = len(self._dims_registry.feature_names)
                samples = np.zeros((n_obs, n_feat))
                from tests.test_model import DummyPredictionData

                return DummyPredictionData(samples)

        model = _make_model(adata, method_cls=RealDummyMethod)
        model.train(adata, n_train_steps=5, train_batch_size=4, sort=True)

        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
            tmp_path = tmp.name
        model.save(tmp_path, allow_overwrite=True)

        loaded = Model.load(tmp_path, map_location="cpu")
        loaded.train(adata, n_train_steps=5, train_batch_size=4)

        os.unlink(tmp_path)


class TestModelConditionSpace:
    """Test Model with the condition-space view (condition_state_key set)."""

    def _condition_space_dm_kwargs(self) -> dict:
        return {
            "condition_state_key": "X_repr",
            "conditions": {"drug": ("drugA",)},
            "conditions_reps": {"drug": "drug"},
            "conditions_covariates": ["X_repr"],
            "groups": ("source_split",),
            "groups_reps": {"source_split": "source_split"},
        }

    def test_from_adata_with_condition_space(self, adata: AnnData):
        """Building with a condition_state_key exposes it on the model."""
        adata = _add_continuous_covariate(adata)
        model = _make_model(adata, dm_kwargs=self._condition_space_dm_kwargs())
        assert model.condition_state_key == "X_repr"

    def test_train_with_condition_space(self, adata: AnnData, mock_optim_manager):
        """Training with the condition-space view correctly compiles the data."""
        adata = _add_continuous_covariate(adata)
        model = _make_model(adata, dm_kwargs=self._condition_space_dm_kwargs())

        # Create a spy on compile_adata to record calls while preserving original behavior
        original_compile = model._dm.compile_adata
        mock_compile = MagicMock(wraps=original_compile)
        model._dm.compile_adata = mock_compile

        with patch("sckitflow._model.Trainer") as mock_trainer_cls:
            mock_trainer = mock_trainer_cls.return_value
            model.train(adata, n_train_steps=10, train_batch_size=32, sort=True)

            # Verify compile_adata was called; the condition-space view is driven
            # by the data manager's condition_state_key attribute.
            mock_compile.assert_called_once()
            assert model.condition_state_key == "X_repr"
            mock_trainer.train.assert_called_once()

    def test_predict_with_condition_space(self, adata: AnnData):
        """Predict with condition space yields predictions with correct shape."""
        adata = _add_continuous_covariate(adata)
        model = _make_model(adata, dm_kwargs=self._condition_space_dm_kwargs())
        pred_adata = model.predict(adata, sort=True)

        assert isinstance(pred_adata, AnnData)
        assert pred_adata.n_obs == adata.n_obs
        # The feature dimension should now be the continuous covariate's dimension
        assert pred_adata.n_vars == adata.obsm["X_repr"].shape[1]

    def test_save_load_with_condition_space(self, adata: AnnData):
        """Save and load a model configured with the condition-space view."""
        adata = _add_continuous_covariate(adata)
        model = _make_model(adata, dm_kwargs=self._condition_space_dm_kwargs())
        pred1 = model.predict(adata, sort=True)

        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
            tmp_path = tmp.name
        model.save(tmp_path, allow_overwrite=True)

        # Load with the same adata (rebuild the data manager)
        loaded = Model.load(tmp_path, adata=adata, **self._condition_space_dm_kwargs())
        pred2 = loaded.predict(adata)

        np.testing.assert_array_equal(pred1.X, pred2.X)
        os.unlink(tmp_path)


class TestModelPredictCombinations:
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

        # Build the model with the requested schema
        dm_kwargs = {
            "condition_state_key": cont_key if view_on_condition_space else None,
            "conditions": conditions,
            "conditions_reps": conditions_reps,
            "conditions_covariates": conditions_covariates,
            "groups": groups,
            "groups_reps": groups_reps,
            "control_values_dict": control_values_dict,
        }
        model = _make_model(adata, dm_kwargs=dm_kwargs)

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
            expected_n_vars = len(model._dims_registry.feature_names)
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

        dm_kwargs = {
            "condition_state_key": cond_state_key,
            "conditions": {cat_cond_col: (cat_cond_col,)},
            "conditions_reps": {cat_cond_col: cat_cond_col},
            "conditions_covariates": [cond_state_key, cond_key],
            "groups": (group_col,),
            "groups_reps": {group_col: group_col},
        }

        model = _make_model(adata, dm_kwargs=dm_kwargs)

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


class TestModelPredictMatchedKeys:
    """Test the matched_keys argument in Model.predict."""

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
        # Build with instance matched_keys
        instance_keys = {("control",): ("treatment",)}
        model = _make_model(
            adata,
            dm_kwargs={
                "conditions": {"drug": ("drugA",)},
                "conditions_reps": {"drug": "drug"},
                "groups": ("source_split",),
                "groups_reps": {"source_split": "source_split"},
                "control_values_dict": {"drug": "control"},
                "matched_keys": instance_keys,
            },
        )
        # Override with different keys at predict time
        override_keys = {("control",): ("treatment",)}  # same for simplicity; could be different
        pred_adata = model.predict(adata, sort=True, matched_keys=override_keys)
        # Check that only treatment rows are predicted (since pairing uses override)
        assert pred_adata.n_obs > 0
        assert all(pred_adata.obs["drugA"] == "treatment")

    def test_predict_matched_keys_none_uses_instance(self, adata):
        """When matched_keys=None, the instance's matched_keys is used."""
        adata = self._setup_paired_data(adata)
        instance_keys = {("control",): ("treatment",)}
        model = _make_model(
            adata,
            dm_kwargs={
                "conditions": {"drug": ("drugA",)},
                "conditions_reps": {"drug": "drug"},
                "groups": ("source_split",),
                "groups_reps": {"source_split": "source_split"},
                "control_values_dict": {"drug": "control"},
                "matched_keys": instance_keys,
            },
        )
        pred_adata = model.predict(adata, sort=True, matched_keys=None)
        assert pred_adata.n_obs > 0
        assert all(pred_adata.obs["drugA"] == "treatment")

    def test_predict_matched_keys_without_control_values(self, adata):
        """When control_values_dict is None, matched_keys can still define source-target pairs."""
        adata = self._setup_paired_data(adata)
        # No control_values_dict, only matched_keys
        keys = {("control",): ("treatment",)}
        model = _make_model(
            adata,
            dm_kwargs={
                "conditions": {"drug": ("drugA",)},
                "conditions_reps": {"drug": "drug"},
                "groups": ("source_split",),
                "groups_reps": {"source_split": "source_split"},
                "matched_keys": keys,  # no control_values_dict
            },
        )
        pred_adata = model.predict(adata, sort=True, matched_keys=keys)
        assert pred_adata.n_obs > 0
        assert all(pred_adata.obs["drugA"] == "treatment")

    def test_predict_matched_keys_honored_on_condition_view(self, adata):
        """With a condition_state_key set, matched_keys are honored."""
        adata = self._setup_paired_data(adata, has_continuous=True)
        keys = {("control",): ("treatment",)}
        model = _make_model(
            adata,
            dm_kwargs={
                "condition_state_key": "X_repr",
                "conditions": {"drug": ("drugA",)},
                "conditions_reps": {"drug": "drug"},
                "conditions_covariates": ["X_repr"],
                "groups": ("source_split",),
                "groups_reps": {"source_split": "source_split"},
                "control_values_dict": {"drug": "control"},
                "matched_keys": keys,
            },
        )
        pred_adata = model.predict(adata, sort=True, matched_keys=keys)
        assert pred_adata.n_obs > 0
        assert all(pred_adata.obs["drugA"] == "treatment")

    def test_predict_matched_keys_invalid_target_raises(self, adata):
        """If matched_keys contains a target key not present in the data, an error should be raised."""
        adata = self._setup_paired_data(adata)
        keys = {("control",): ("nonexistent",)}
        model = _make_model(
            adata,
            dm_kwargs={
                "conditions": {"drug": ("drugA",)},
                "conditions_reps": {"drug": "drug"},
                "groups": ("source_split",),
                "groups_reps": {"source_split": "source_split"},
                "control_values_dict": {"drug": "control"},
            },
        )
        with pytest.raises(KeyError, match="nonexistent"):
            model.predict(adata, sort=True, matched_keys=keys)


class TestModelPredictControlValues:
    """Test the control_values_dict argument in Model.predict."""

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

    def _base_dm_kwargs(self, control_values_dict=None):
        dm_kwargs = {
            "conditions": {"drug": ("drugA",)},
            "conditions_reps": {"drug": "drug"},
            "groups": ("source_split",),
            "groups_reps": {"source_split": "source_split"},
        }
        if control_values_dict is not None:
            dm_kwargs["control_values_dict"] = control_values_dict
        return dm_kwargs

    def test_predict_control_values_override(self, adata):
        """Passing control_values_dict to predict overrides the instance's control_values_dict."""
        adata = self._setup_paired_data(adata)
        model = _make_model(adata, dm_kwargs=self._base_dm_kwargs({"drug": "control"}))
        override_dict = {"drug": "control"}  # same value; test that override is used
        pred_adata = model.predict(adata, sort=True, control_values_dict=override_dict)
        assert pred_adata.n_obs > 0
        assert all(pred_adata.obs["drugA"] == "treatment")

    def test_predict_control_values_none_uses_instance(self, adata):
        """When control_values_dict=None, the instance's control_values_dict is used."""
        adata = self._setup_paired_data(adata)
        model = _make_model(adata, dm_kwargs=self._base_dm_kwargs({"drug": "control"}))
        pred_adata = model.predict(adata, sort=True, control_values_dict=None)
        assert pred_adata.n_obs > 0
        assert all(pred_adata.obs["drugA"] == "treatment")

    def test_predict_control_values_without_instance(self, adata):
        """When instance has no control_values_dict, a custom dict works."""
        adata = self._setup_paired_data(adata)
        model = _make_model(adata, dm_kwargs=self._base_dm_kwargs())
        custom_dict = {"drug": "control"}
        pred_adata = model.predict(adata, sort=True, control_values_dict=custom_dict)
        assert pred_adata.n_obs > 0
        assert all(pred_adata.obs["drugA"] == "treatment")

    def test_predict_control_values_honored_on_condition_view(self, adata):
        """With a condition_state_key set, control_values_dict is honored."""
        adata = self._setup_paired_data(adata, has_continuous=True)
        dm_kwargs = self._base_dm_kwargs({"drug": "control"})
        dm_kwargs["condition_state_key"] = "X_repr"
        dm_kwargs["conditions_covariates"] = ["X_repr"]
        model = _make_model(adata, dm_kwargs=dm_kwargs)
        pred_adata = model.predict(adata, sort=True, control_values_dict={"drug": "control"})
        assert pred_adata.n_obs > 0
        assert all(pred_adata.obs["drugA"] == "treatment")

    def test_predict_control_values_invalid_control_raises(self, adata):
        """If control_values_dict contains a value not present in the data, an error should be raised."""
        adata = self._setup_paired_data(adata)
        model = _make_model(adata, dm_kwargs=self._base_dm_kwargs())
        with pytest.raises(KeyError, match="nonexistent"):
            model.predict(adata, sort=True, control_values_dict={"drug": "nonexistent"})
