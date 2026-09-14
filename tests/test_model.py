import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch
from anndata import AnnData
from tests.data.shared import with_split

from sckitflow import Model, ModelBuilder
from sckitflow.core.methods._base import (
    BaseInferenceProtocol,
    BaseTrainingProtocol,
    MatchedTrainingProtocol,
    SupportsInference,
    SupportsProtocol,
    SupportsTraining,
)
from sckitflow.core.nn._modules import BaseModule
from sckitflow.data._manager import DataManager


# -----------------------------------------------------------------------------
# Dummy module: picklable, exposes `n_features`, supports `init_from_dims_registry`,
# and implements `_make_modules` (abstract in `BaseModule`).
# -----------------------------------------------------------------------------
class DummyModule(BaseModule):
    def __init__(self, n_features: int = 10):
        super().__init__()
        self.n_features = n_features

    def _make_modules(self, *args, **kwargs):
        pass

    @classmethod
    def init_from_dims_registry(cls, dims_registry, n_features=None, **kwargs):
        if n_features is None:
            n_features = len(dims_registry.feature_names)
        return cls(n_features=n_features)

    def forward(self, *args, **kwargs):
        pass


# -----------------------------------------------------------------------------
# Dummy PredictionData: Model calls `type(all_preds[0]).concatenate(all_preds)`
# when aggregating, so the classmethod is required.
# -----------------------------------------------------------------------------
class DummyPredictionData:
    def __init__(self, X, traj=None, raw_samples=None):
        self.X = X
        self.traj = traj
        self.raw_samples = raw_samples

    @classmethod
    def concatenate(cls, preds):
        X = np.concatenate([p.X for p in preds], axis=0)
        trajs = [p.traj for p in preds if p.traj is not None]
        traj = np.concatenate(trajs, axis=1) if trajs else None
        return cls(X, traj)


# -----------------------------------------------------------------------------
# Dummy protocols
# -----------------------------------------------------------------------------
class DummyTrainingProtocol(BaseTrainingProtocol):
    """Concrete training protocol; the loss is a constant."""

    def compute_loss(self, step_data, *args, **kwargs):
        return torch.tensor(0.0), {"loss": 0.0}


class DummyInferenceProtocol(BaseInferenceProtocol):
    """Concrete inference protocol sizing the output from the module's `n_features`."""

    def predict(self, step_data, *args, **kwargs):
        n_feat = self.module.n_features
        if step_data["target_state"] is not None:
            n_obs = step_data["target_state"].shape[0]
        else:
            cond = step_data["target_condition_data"] or step_data["target_group_data"] or {}
            n_obs = next(iter(cond.values())).shape[0]
        return DummyPredictionData(np.zeros((n_obs, n_feat)))


# -----------------------------------------------------------------------------
# Dummy match functions. Module-level so they satisfy `Model.save` picklability
# (cloudpickle can serialize top-level functions).
# -----------------------------------------------------------------------------
def dummy_match_fn(source_lin=None, target_lin=None, source_quad=None, target_quad=None):
    """No-op matcher: returns no indices so `MatchingProtocol.match` short-circuits."""
    return None, None


def other_match_fn(source_lin=None, target_lin=None, source_quad=None, target_quad=None):
    """A distinct no-op matcher for override tests."""
    return None, None


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _add_continuous_covariate(adata: AnnData, key: str = "X_repr", n_dim: int = 10) -> AnnData:
    """Add a random continuous covariate to `adata.obsm`."""
    adata.obsm[key] = np.random.randn(adata.n_obs, n_dim)
    return adata


def _make_model(
    adata: AnnData,
    dm_kwargs: dict | None = None,
    module_cls: type | None = DummyModule,
    training_protocol_cls: type | None = DummyTrainingProtocol,
    inference_protocol_cls: type | None = DummyInferenceProtocol,
    **overrides,
) -> Model:
    """Build a Model with the dummy protocols by default; any can be overridden."""
    builder = ModelBuilder.from_adata(adata, **(dm_kwargs or {}))
    return builder.build(
        module_cls=module_cls,
        training_protocol_cls=training_protocol_cls,
        inference_protocol_cls=inference_protocol_cls,
        **overrides,
    )


_DM_TRAIN_KWARGS = {
    "conditions": {"drug": ("drugA",)},
    "conditions_reps": {"drug": "drug"},
    "groups": ("source_split",),
    "groups_reps": {"source_split": "source_split"},
    "split_by": "split",
}


def _with_split(adata: AnnData, labels=("train", "val1", "val2")) -> AnnData:
    """Attach a `split` column, each `(source_split, drugA)` group wholly in one split."""
    return with_split(adata, cols=("source_split", "drugA"), labels=labels)


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
    # ------------------------------------------------------------------
    # Construction and initialization
    # ------------------------------------------------------------------
    def test_builder_builds_dm_and_dims(self, adata: AnnData):
        model = _make_model(adata)
        assert isinstance(model.dm, DataManager)
        assert model._dims_registry is not None
        assert model.is_paired_setting is False
        assert len(model._dims_registry.feature_names) == adata.n_vars

    def test_builder_exposes_dm_and_dims(self, adata: AnnData):
        builder = ModelBuilder.from_adata(adata)
        assert isinstance(builder.dm, DataManager)
        assert builder.data_dims is not None
        assert len(builder.data_dims.feature_names) == adata.n_vars

    def test_build_with_module_instance(self, adata: AnnData):
        """`build(module=...)` attaches a pre-built module verbatim."""
        builder = ModelBuilder.from_adata(adata)
        module = DummyModule(n_features=adata.n_vars)
        model = builder.build(
            module=module,
            training_protocol_cls=DummyTrainingProtocol,
            inference_protocol_cls=DummyInferenceProtocol,
        )
        assert model.module is module

    def test_control_key_sets_paired_true(self, adata: AnnData):
        model = _make_model(adata, dm_kwargs={"control_values_dict": {"drugA": "control"}})
        assert model.is_paired_setting is True

    def test_direct_init_from_dm_and_dims(self, adata: AnnData):
        """Model can be constructed directly from a data manager and dimensionalities."""
        dm = DataManager()
        data_dims = dm.get_data_dimensionalities(adata)
        model = Model(
            dm,
            data_dims,
            module_cls=DummyModule,
            training_protocol_cls=DummyTrainingProtocol,
            inference_protocol_cls=DummyInferenceProtocol,
        )
        assert model.dm is dm
        assert model._dims_registry is data_dims

    def test_init_raises_without_protocols(self, adata: AnnData):
        builder = ModelBuilder.from_adata(adata)
        with pytest.raises(ValueError, match="At least one of"):
            builder.build(module_cls=DummyModule)

    # ------------------------------------------------------------------
    # Structural contracts on the constructed protocols
    # ------------------------------------------------------------------
    def test_default_protocols_satisfy_structural_contracts(self, adata: AnnData):
        """Unmatched construction still yields `SupportsTraining` / `SupportsInference`."""
        model = _make_model(adata)
        assert isinstance(model.training_protocol, SupportsTraining)
        assert isinstance(model.training_protocol, SupportsProtocol)
        assert isinstance(model.inference_protocol, SupportsInference)
        assert isinstance(model.inference_protocol, SupportsProtocol)

    def test_training_protocol_does_not_satisfy_inference_contract(self, adata: AnnData):
        """A training-only protocol has no `predict`, so it does not satisfy `SupportsInference`."""
        model = _make_model(adata)
        assert not isinstance(model.training_protocol, SupportsInference)

    def test_inference_protocol_does_not_satisfy_training_contract(self, adata: AnnData):
        """An inference-only protocol has no `compute_loss`, so it does not satisfy `SupportsTraining`."""
        model = _make_model(adata)
        assert not isinstance(model.inference_protocol, SupportsTraining)

    # ------------------------------------------------------------------
    # Protocol resolution
    #
    # `_model.py` binds the registries locally via
    # `from sckitflow.core.methods import TRAINING_PROTOCOLS_REGISTRY,
    #  INFERENCE_PROTOCOLS_REGISTRY`. Patching must target the local
    # binding, not the definition site.
    # ------------------------------------------------------------------
    def test_training_protocol_id_resolves_to_registered_class(self, adata: AnnData, monkeypatch):
        monkeypatch.setattr(
            "sckitflow._model.TRAINING_PROTOCOLS_REGISTRY",
            {"cfm": DummyTrainingProtocol},
        )
        builder = ModelBuilder.from_adata(adata)
        model = builder.build(
            module_cls=DummyModule,
            training_protocol_id="cfm",
            inference_protocol_cls=DummyInferenceProtocol,
        )
        assert isinstance(model.training_protocol, DummyTrainingProtocol)

    def test_inference_protocol_id_resolves_to_registered_class(self, adata: AnnData, monkeypatch):
        monkeypatch.setattr(
            "sckitflow._model.INFERENCE_PROTOCOLS_REGISTRY",
            {"ode": DummyInferenceProtocol},
        )
        builder = ModelBuilder.from_adata(adata)
        model = builder.build(
            module_cls=DummyModule,
            training_protocol_cls=DummyTrainingProtocol,
            inference_protocol_id="ode",
        )
        assert isinstance(model.inference_protocol, DummyInferenceProtocol)

    def test_unsupported_training_protocol_id_raises(self, adata: AnnData, monkeypatch):
        monkeypatch.setattr(
            "sckitflow._model.TRAINING_PROTOCOLS_REGISTRY",
            {"cfm": DummyTrainingProtocol},
        )
        builder = ModelBuilder.from_adata(adata)
        with pytest.raises(KeyError):
            builder.build(
                module_cls=DummyModule,
                training_protocol_id="does_not_exist",
                inference_protocol_cls=DummyInferenceProtocol,
            )

    # ------------------------------------------------------------------
    # _to_numpy
    # ------------------------------------------------------------------
    def test_to_numpy_with_torch_tensor(self, adata: AnnData):
        model = _make_model(adata)
        t = torch.tensor([1.0, 2.0, 3.0])
        arr = model._to_numpy(t)
        assert isinstance(arr, np.ndarray)
        np.testing.assert_array_equal(arr, np.array([1.0, 2.0, 3.0]))

    def test_to_numpy_returns_none_for_none(self, adata: AnnData):
        model = _make_model(adata)
        assert model._to_numpy(None) is None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def test_train_calls_trainer_and_sets_mode(self, adata: AnnData, mock_optim_manager):
        adata = _with_split(adata)
        model = _make_model(adata, dm_kwargs=_DM_TRAIN_KWARGS)
        with patch("sckitflow._model.Trainer") as mock_trainer_cls:
            mock_trainer = mock_trainer_cls.return_value
            spy = MagicMock(wraps=model.training_protocol.set_train_mode)
            model.training_protocol.set_train_mode = spy

            model.train(adata, n_train_steps=10, valid_freq=5, batch_size=32)

            mock_trainer_cls.assert_called_once()
            args, kwargs = mock_trainer_cls.call_args
            assert args[0] is model.training_protocol
            assert kwargs.get("inference_protocol") is model.inference_protocol
            spy.assert_called_once_with(True)
            mock_trainer.train.assert_called_once()
            train_args, train_kwargs = mock_trainer.train.call_args
            assert len(train_args[0]) == 10
            assert train_kwargs["valid_freq"] == 5

    def test_train_with_validation_loaders(self, adata: AnnData, mock_optim_manager):
        adata = _with_split(adata)
        model = _make_model(adata, dm_kwargs=_DM_TRAIN_KWARGS)
        with patch("sckitflow._model.Trainer") as mock_trainer_cls:
            mock_trainer = mock_trainer_cls.return_value
            model.train(adata, n_train_steps=5)
            call_kwargs = mock_trainer.train.call_args.kwargs
            val_loaders = call_kwargs["val_loaders"]
            assert isinstance(val_loaders, dict)
            assert set(val_loaders.keys()) == {"val1", "val2"}

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def test_predict_returns_anndata_with_correct_shape(self, adata: AnnData):
        model = _make_model(adata, dm_kwargs=_DM_TRAIN_KWARGS)
        pred_adata = model.predict(adata)
        assert isinstance(pred_adata, AnnData)
        assert pred_adata.n_obs == adata.n_obs
        assert pred_adata.n_vars == adata.n_vars
        assert set(pred_adata.obs.columns) == {"source_split", "drugA"}

    def test_predict_without_target_state(self, adata: AnnData):
        """predict(require_target_state=False, max_per_group=1) works with metadata only."""
        dm_kwargs = {**_DM_TRAIN_KWARGS, "conditions_covariates": ["X_repr"]}
        model = _make_model(adata, dm_kwargs=dm_kwargs)

        meta = AnnData(obs=adata.obs[["source_split", "drugA"]].copy())
        meta.uns = dict(adata.uns)
        meta.obsm["X_repr"] = np.random.randn(meta.n_obs, 8).astype(np.float32)

        pred_adata = model.predict(meta, require_target_state=False, max_per_group=1)

        assert isinstance(pred_adata, AnnData)
        n_groups = meta.obs.astype(str).drop_duplicates().shape[0]
        assert pred_adata.n_obs == n_groups
        assert pred_adata.n_vars == adata.n_vars
        assert "X_repr" in pred_adata.obsm

    def test_predict_with_return_raw(self, adata: AnnData):
        model = _make_model(adata, dm_kwargs=_DM_TRAIN_KWARGS)
        result = model.predict(adata, return_raw=True)
        assert isinstance(result, tuple) and len(result) == 2
        pred_adata, pred_data = result
        assert isinstance(pred_adata, AnnData)
        assert hasattr(pred_data, "X") and hasattr(pred_data, "traj")

    def test_predict_empty_returns_empty_anndata(self, adata: AnnData, monkeypatch):
        model = _make_model(adata, dm_kwargs=_DM_TRAIN_KWARGS)

        class _EmptyEval:
            group_cols = ()
            cond_cont_keys = ()
            resp_keys = ()

            def __len__(self):
                return 0

            def __iter__(self):
                return iter(())

        monkeypatch.setattr(model._dm, "get_eval_loader", lambda *a, **k: _EmptyEval())
        pred_adata = model.predict(adata)
        assert pred_adata.n_obs == 0
        assert pred_adata.n_vars == adata.n_vars

    def test_predict_sets_eval_mode(self, adata: AnnData):
        model = _make_model(adata, dm_kwargs=_DM_TRAIN_KWARGS)
        spy = MagicMock(wraps=model.inference_protocol.set_train_mode)
        model.inference_protocol.set_train_mode = spy
        model.predict(adata)
        spy.assert_called_once_with(False)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    def test_properties(self, adata: AnnData, mock_optim_manager):
        model = _make_model(adata)
        assert isinstance(model.dm, DataManager)
        assert model.is_paired_setting is False
        # Structural — not nominal — the properties are typed `Supports*`.
        assert isinstance(model.training_protocol, SupportsTraining)
        assert isinstance(model.inference_protocol, SupportsInference)
        # Concrete classes still satisfy the structural contracts too.
        assert isinstance(model.training_protocol, BaseTrainingProtocol)
        assert isinstance(model.inference_protocol, BaseInferenceProtocol)
        assert model.trainer is None
        assert model.condition_state_key is None

        adata = _with_split(adata)
        model = _make_model(adata, dm_kwargs=_DM_TRAIN_KWARGS)
        with patch("sckitflow._model.Trainer"):
            model.train(adata, n_train_steps=1)
        assert model.trainer is not None

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------
    def test_save_load_and_predict(self, adata):
        model = _make_model(adata, dm_kwargs=_DM_TRAIN_KWARGS)
        pred1 = model.predict(adata)

        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
            tmp_path = tmp.name
        model.save(tmp_path, allow_overwrite=True)

        loaded = Model.load(tmp_path, map_location="cpu")
        pred2 = loaded.predict(adata)

        np.testing.assert_array_equal(pred1.X, pred2.X)
        os.unlink(tmp_path)

    def test_save_load_and_continue_training(self, adata):
        """Save/load with a real tiny module, training end-to-end through the loader."""
        adata = _with_split(adata)

        class RealDummyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)

            def forward(self, t, x, condition_dict=None, source=None):
                return self.linear(x)

            @classmethod
            def init_from_dims_registry(cls, dims_registry, *args, **kwargs):
                return cls()

        class RealDummyTrainingProtocol(BaseTrainingProtocol):
            def set_train_mode(self, mode: bool):
                self.module.train() if mode else self.module.eval()

            def compute_loss(self, step_data, *args, **kwargs):
                dummy_x = torch.randn(4, 2, requires_grad=True)
                t = torch.tensor(0.5)
                loss = self.module(t, dummy_x).sum()
                return loss, {"loss": loss.item()}

        class RealDummyInferenceProtocol(BaseInferenceProtocol):
            def predict(self, step_data, *args, **kwargs):
                return DummyPredictionData(np.zeros((1, 2)))

        model = _make_model(
            adata,
            dm_kwargs=_DM_TRAIN_KWARGS,
            module_cls=RealDummyModule,
            training_protocol_cls=RealDummyTrainingProtocol,
            inference_protocol_cls=RealDummyInferenceProtocol,
        )
        model.train(adata, n_train_steps=5, batch_size=4)

        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
            tmp_path = tmp.name
        model.save(tmp_path, allow_overwrite=True)

        loaded = Model.load(tmp_path, map_location="cpu")
        loaded.train(adata, n_train_steps=5, batch_size=4)

        os.unlink(tmp_path)


# -----------------------------------------------------------------------------
# match_fn integration
# -----------------------------------------------------------------------------
class TestModelMatching:
    """`match_fn` wraps the training protocol in `MatchedTrainingProtocol`."""

    def test_construction_with_match_fn_wraps_training_protocol(self, adata):
        model = _make_model(adata, match_fn=dummy_match_fn)
        assert isinstance(model.training_protocol, MatchedTrainingProtocol)
        assert model.training_protocol.matcher.match_fn is dummy_match_fn

    def test_construction_without_match_fn_leaves_protocol_unwrapped(self, adata):
        model = _make_model(adata)
        assert not isinstance(model.training_protocol, MatchedTrainingProtocol)
        assert isinstance(model.training_protocol, DummyTrainingProtocol)

    def test_matched_protocol_satisfies_structural_contract(self, adata):
        """The wrapper satisfies `SupportsTraining` even though it does not subclass `BaseTrainingProtocol`."""
        model = _make_model(adata, match_fn=dummy_match_fn)
        assert isinstance(model.training_protocol, SupportsTraining)
        assert not isinstance(model.training_protocol, BaseTrainingProtocol)

    def test_train_without_per_call_match_fn_keeps_construction_matcher(self, adata, mock_optim_manager):
        """A `train()` call with no `match_fn` uses the instance's (already matched) protocol."""
        adata = _with_split(adata)
        model = _make_model(adata, dm_kwargs=_DM_TRAIN_KWARGS, match_fn=dummy_match_fn)
        assert isinstance(model.training_protocol, MatchedTrainingProtocol)

        with patch("sckitflow._model.Trainer") as mock_trainer_cls:
            model.train(adata, n_train_steps=2)
            training_arg = mock_trainer_cls.call_args[0][0]
            # The instance's matched protocol is passed through unchanged.
            assert training_arg is model.training_protocol
            assert isinstance(training_arg, MatchedTrainingProtocol)
            assert training_arg.matcher.match_fn is dummy_match_fn

    def test_train_per_call_match_fn_wraps_unmatched_protocol(self, adata, mock_optim_manager):
        """A `train(match_fn=...)` call wraps an unmatched instance protocol for that call only."""
        adata = _with_split(adata)
        model = _make_model(adata, dm_kwargs=_DM_TRAIN_KWARGS)
        original = model.training_protocol
        assert not isinstance(original, MatchedTrainingProtocol)

        with patch("sckitflow._model.Trainer") as mock_trainer_cls:
            model.train(adata, n_train_steps=2, match_fn=dummy_match_fn)
            training_arg = mock_trainer_cls.call_args[0][0]
            assert isinstance(training_arg, MatchedTrainingProtocol)
            assert training_arg.matcher.match_fn is dummy_match_fn
            # The instance's stored protocol is untouched.
            assert model.training_protocol is original
            assert not isinstance(model.training_protocol, MatchedTrainingProtocol)

    def test_train_per_call_match_fn_overrides_construction_matcher(self, adata, mock_optim_manager):
        """A per-call `match_fn` replaces the construction-time matcher for that call only."""
        adata = _with_split(adata)
        model = _make_model(adata, dm_kwargs=_DM_TRAIN_KWARGS, match_fn=dummy_match_fn)
        original = model.training_protocol

        with patch("sckitflow._model.Trainer") as mock_trainer_cls:
            model.train(adata, n_train_steps=2, match_fn=other_match_fn)
            training_arg = mock_trainer_cls.call_args[0][0]
            assert isinstance(training_arg, MatchedTrainingProtocol)
            assert training_arg.matcher.match_fn is other_match_fn
            # The stored protocol still carries the construction-time matcher.
            assert model.training_protocol is original
            assert model.training_protocol.matcher.match_fn is dummy_match_fn

    def test_train_per_call_protocol_without_match_fn_is_unwrapped(self, adata, mock_optim_manager):
        """A per-call protocol override with no `match_fn` is used raw."""
        adata = _with_split(adata)
        model = _make_model(adata, dm_kwargs=_DM_TRAIN_KWARGS, match_fn=dummy_match_fn)

        with patch("sckitflow._model.Trainer") as mock_trainer_cls:
            model.train(
                adata,
                n_train_steps=2,
                training_protocol_cls=DummyTrainingProtocol,
            )
            training_arg = mock_trainer_cls.call_args[0][0]
            # The per-call override bypasses construction-time matching.
            assert not isinstance(training_arg, MatchedTrainingProtocol)
            assert isinstance(training_arg, DummyTrainingProtocol)

    def test_save_load_preserves_matching(self, adata):
        """A model saved with `match_fn` reloads with the matcher intact."""
        model = _make_model(adata, dm_kwargs=_DM_TRAIN_KWARGS, match_fn=dummy_match_fn)
        assert isinstance(model.training_protocol, MatchedTrainingProtocol)

        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
            tmp_path = tmp.name
        model.save(tmp_path, allow_overwrite=True)

        loaded = Model.load(tmp_path, map_location="cpu")
        assert isinstance(loaded.training_protocol, MatchedTrainingProtocol)
        # The matcher's callable survives pickling (dummy_match_fn is module-level).
        assert loaded.training_protocol.matcher.match_fn is dummy_match_fn

        os.unlink(tmp_path)


class TestModelConditionSpace:
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
        adata = _add_continuous_covariate(adata)
        model = _make_model(adata, dm_kwargs=self._condition_space_dm_kwargs())
        assert model.condition_state_key == "X_repr"

    def test_train_with_condition_space(self, adata: AnnData, mock_optim_manager):
        adata = _add_continuous_covariate(adata)
        adata = _with_split(adata)
        model = _make_model(
            adata,
            dm_kwargs={**self._condition_space_dm_kwargs(), "split_by": "split"},
        )
        spy = MagicMock(wraps=model._dm.get_dataloaders)
        model._dm.get_dataloaders = spy

        with patch("sckitflow._model.Trainer") as mock_trainer_cls:
            mock_trainer = mock_trainer_cls.return_value
            model.train(adata, n_train_steps=10, batch_size=32)

            spy.assert_called_once()
            assert model.condition_state_key == "X_repr"
            mock_trainer.train.assert_called_once()

    def test_predict_with_condition_space(self, adata: AnnData):
        adata = _add_continuous_covariate(adata)
        model = _make_model(adata, dm_kwargs=self._condition_space_dm_kwargs())
        pred_adata = model.predict(adata)

        assert isinstance(pred_adata, AnnData)
        assert pred_adata.n_obs == adata.n_obs
        assert pred_adata.n_vars == adata.obsm["X_repr"].shape[1]

    def test_save_load_with_condition_space(self, adata: AnnData):
        adata = _add_continuous_covariate(adata)
        model = _make_model(adata, dm_kwargs=self._condition_space_dm_kwargs())
        pred1 = model.predict(adata)

        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
            tmp_path = tmp.name
        model.save(tmp_path, allow_overwrite=True)

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
        if view_on_condition_space and not has_cont_cond:
            pytest.skip("view_on_condition_space requires a continuous condition covariate")
        if not (has_cat_cond or has_groups or has_source):
            pytest.skip("EvalLoader requires a categorical group/condition column")

        adata = adata.copy()
        base_n_obs = adata.n_obs

        keep_cols = []
        if has_cat_cond:
            keep_cols.append("drugA")
        if has_groups:
            keep_cols.append("source_split")
        adata.obs = adata.obs[keep_cols].copy() if keep_cols else pd.DataFrame(index=adata.obs_names)

        conditions = {}
        conditions_reps = {}
        conditions_covariates = [] if has_cont_cond else None
        groups = None
        groups_reps = {}
        control_values_dict = None

        if has_cat_cond:
            realm_col = "drug"
            cat_col = "drugA"
            control_val = "control"
            unique_vals = adata.obs[cat_col].unique()
            rep_dim = 4
            adata.uns[realm_col] = {v: np.random.randn(rep_dim) for v in unique_vals}
            if has_source:
                if control_val not in adata.uns[realm_col]:
                    adata.uns[realm_col][control_val] = np.random.randn(rep_dim)
                control_values_dict = {realm_col: control_val}
            conditions[realm_col] = (cat_col,)
            conditions_reps[realm_col] = realm_col

        cont_key = "X_repr"
        if has_cont_cond:
            adata = _add_continuous_covariate(adata, key=cont_key, n_dim=5)
            conditions_covariates = [cont_key]

        if has_groups:
            group_col = "source_split"
            groups = (group_col,)
            groups_reps[group_col] = group_col
            unique_groups = adata.obs[group_col].unique()
            adata.uns[group_col] = {v: np.random.randn(2) for v in unique_groups}

        if has_source and not has_cat_cond:
            dummy_col = "dummy_paired"
            n_obs = len(adata)
            n_control = n_obs // 2
            adata.obs[dummy_col] = ["control"] * n_control + ["treatment"] * (n_obs - n_control)
            adata.obs[dummy_col] = adata.obs[dummy_col].astype("category")
            conditions[dummy_col] = (dummy_col,)
            conditions_reps[dummy_col] = dummy_col
            adata.uns[dummy_col] = {
                "control": np.random.randn(2),
                "treatment": np.random.randn(2),
            }
            control_values_dict = {dummy_col: "control"}

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

        captured_step_data = []
        original_predict = model.inference_protocol.predict

        def spy_predict(step_data, *args, **kwargs):
            captured_step_data.append(step_data)
            return original_predict(step_data, *args, **kwargs)

        model.inference_protocol.predict = spy_predict

        pred_adata = model.predict(adata)

        if has_source:
            control_dict = model._dm.control_values_dict
            cond_schema = model._dm.condition_data_schema
            realm = next(iter(control_dict.keys()))
            control_val = control_dict[realm]
            col = cond_schema.conditions[realm][0]
            expected_n_obs = (adata.obs[col] != control_val).sum()
        else:
            expected_n_obs = base_n_obs

        assert pred_adata.n_obs == expected_n_obs

        if view_on_condition_space:
            expected_n_vars = adata.obsm[cont_key].shape[1]
        else:
            expected_n_vars = len(model._dims_registry.feature_names)
        assert pred_adata.n_vars == expected_n_vars

        step_data = captured_step_data[0]
        if has_groups:
            assert step_data["target_group_data"] is not None
            assert group_col in step_data["target_group_data"]
        if has_cat_cond:
            assert step_data["target_condition_data"] is not None
            assert realm_col in step_data["target_condition_data"]

        assert (step_data["source_state"] is not None) == has_source

    def test_continuous_covariates_flow_to_step_data_and_obsm(self, adata):
        """Continuous condition covariates ride per-cell into StepData and out to obsm."""
        adata = adata.copy()
        cond_key = "paired_condition"
        adata.obsm[cond_key] = np.random.randn(adata.n_obs, 3).astype(np.float32)

        cat_cond_col = "drugA"
        group_col = "source_split"
        adata.obs = adata.obs[[cat_cond_col, group_col]].copy()
        adata.uns[cat_cond_col] = {v: np.random.randn(2) for v in adata.obs[cat_cond_col].unique()}
        adata.uns[group_col] = {v: np.random.randn(2) for v in adata.obs[group_col].unique()}

        dm_kwargs = {
            "conditions": {cat_cond_col: (cat_cond_col,)},
            "conditions_reps": {cat_cond_col: cat_cond_col},
            "conditions_covariates": [cond_key],
            "groups": (group_col,),
            "groups_reps": {group_col: group_col},
        }
        model = _make_model(adata, dm_kwargs=dm_kwargs)

        captured = []
        original_predict = model.inference_protocol.predict

        def spy(step_data, *args, **kwargs):
            captured.append(step_data)
            return original_predict(step_data, *args, **kwargs)

        model.inference_protocol.predict = spy

        pred = model.predict(adata)

        for step_data in captured:
            cond = step_data["target_condition_data"]
            assert cond is not None
            assert cat_cond_col in cond
            assert cond_key in cond
            assert step_data["target_group_data"] is not None
            assert group_col in step_data["target_group_data"]

        assert cond_key in pred.obsm
        assert pred.obsm[cond_key].shape[0] == pred.n_obs


class TestModelPredictControlValues:
    """Test the `control_values_dict` argument in `Model.predict`."""

    def _setup_paired_data(self, adata, has_continuous=False):
        adata = adata.copy()
        adata.obs = adata.obs[["drugA", "source_split"]].copy()
        adata.obs["drugA"] = adata.obs["drugA"].astype(str)
        n_obs = len(adata)
        control_vals = ["control"] * (n_obs // 2)
        treatment_vals = ["treatment"] * (n_obs - n_obs // 2)
        adata.obs["drugA"] = control_vals + treatment_vals
        adata.uns["drug"] = {
            "control": np.random.randn(4),
            "treatment": np.random.randn(4),
        }
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
        adata = self._setup_paired_data(adata)
        model = _make_model(adata, dm_kwargs=self._base_dm_kwargs({"drug": "control"}))
        pred_adata = model.predict(adata, control_values_dict={"drug": "control"})
        assert pred_adata.n_obs > 0
        assert all(pred_adata.obs["drugA"] == "treatment")

    def test_predict_control_values_none_uses_instance(self, adata):
        adata = self._setup_paired_data(adata)
        model = _make_model(adata, dm_kwargs=self._base_dm_kwargs({"drug": "control"}))
        pred_adata = model.predict(adata, control_values_dict=None)
        assert pred_adata.n_obs > 0
        assert all(pred_adata.obs["drugA"] == "treatment")

    def test_predict_control_values_without_instance(self, adata):
        adata = self._setup_paired_data(adata)
        model = _make_model(adata, dm_kwargs=self._base_dm_kwargs())
        pred_adata = model.predict(adata, control_values_dict={"drug": "control"})
        assert pred_adata.n_obs > 0
        assert all(pred_adata.obs["drugA"] == "treatment")

    def test_predict_control_values_honored_on_condition_view(self, adata):
        adata = self._setup_paired_data(adata, has_continuous=True)
        dm_kwargs = self._base_dm_kwargs({"drug": "control"})
        dm_kwargs["condition_state_key"] = "X_repr"
        dm_kwargs["conditions_covariates"] = ["X_repr"]
        model = _make_model(adata, dm_kwargs=dm_kwargs)
        pred_adata = model.predict(adata, control_values_dict={"drug": "control"})
        assert pred_adata.n_obs > 0
        assert all(pred_adata.obs["drugA"] == "treatment")

    def test_predict_control_values_invalid_control_predicts_all(self, adata):
        adata = self._setup_paired_data(adata)
        model = _make_model(adata, dm_kwargs=self._base_dm_kwargs())
        pred_adata = model.predict(adata, control_values_dict={"drug": "nonexistent"})
        assert set(pred_adata.obs["drugA"].unique()) == {"control", "treatment"}
