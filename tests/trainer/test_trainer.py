# tests/trainer/test_trainer.py
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
import torch

from sckitflow.core.methods._base import BaseInferenceProtocol, BaseTrainingProtocol
from sckitflow.core.methods._opt import OptimizationManager
from sckitflow.core.nn._modules import BaseModule
from sckitflow.trainer._callbacks import ComputationalCallback, LoggingCallback
from sckitflow.trainer._trainer import Trainer


# -----------------------------------------------------------------------------
# Dummy module
# -----------------------------------------------------------------------------
class DummyModule(BaseModule):
    """Minimal module with parameters, enough to satisfy `ProtocolSpecs.__init__`."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)

    def _make_modules(self, dims_registry=None, *args, **kwargs):
        # BaseModule may call this during setup; keep it a no-op.
        pass

    def forward(self, t, x, condition_dict=None, source=None):
        return self.linear(x)


class DummyPredictionData:
    """Minimal stand-in for the real ``PredictionData``; the Trainer reads ``.X``."""

    def __init__(self, X, traj=None, raw_samples=None):
        self.X = X
        self.traj = traj
        self.raw_samples = raw_samples


# -----------------------------------------------------------------------------
# Dummy protocols
# -----------------------------------------------------------------------------
class DummyTrainingProtocol(BaseTrainingProtocol):
    """Concrete training protocol: a constant loss and metric dict."""

    def compute_loss(self, step_data, *args, **kwargs):
        return 0.5, {"loss": 0.5, "accuracy": 0.8}


class DummyInferenceProtocol(BaseInferenceProtocol):
    """Concrete inference protocol: returns a `PredictionData` with `.X`."""

    def predict(self, step_data, *args, **kwargs):
        return DummyPredictionData(np.random.randn(10, 5), traj=None, raw_samples=None)


# -----------------------------------------------------------------------------
# Dummy optimizer manager
# -----------------------------------------------------------------------------
class DummyOptManager(OptimizationManager):
    def __init__(self):
        super().__init__(None, None, None)

    def step(self, loss):
        pass


# -----------------------------------------------------------------------------
# Dummy loaders
# -----------------------------------------------------------------------------
class DummyTrainLoader:
    """Finite, re-iterable loader yielding `n` `Mock` StepData batches."""

    def __init__(self, n=2):
        self._n = n

    def __iter__(self):
        return iter([Mock() for _ in range(self._n)])


class DummyValLoader:
    """Yields two `MagicMock` batches (subscriptable, so `batch["target_state"]` works)."""

    def __iter__(self):
        return iter([MagicMock(), MagicMock()])


# -----------------------------------------------------------------------------
# Recording callbacks
# -----------------------------------------------------------------------------
class RecordingCallback(LoggingCallback):
    """Logging callback; `TrainingCallbacks` forwards train hooks here."""

    def __init__(self):
        self.train_begin_calls = []
        self.train_step_calls = []
        self.valid_step_calls = []
        self.train_end_calls = []

    def on_train_begin(self, trainer, **kwargs):
        self.train_begin_calls.append((trainer, kwargs))

    def on_train_step(self, trainer, step, logs, **kwargs):
        self.train_step_calls.append((trainer, step, logs, kwargs))

    def on_valid_step(self, trainer, step, val_id, predictions_dict, **kwargs):
        self.valid_step_calls.append((trainer, step, val_id, predictions_dict, kwargs))

    def on_train_end(self, trainer, **kwargs):
        self.train_end_calls.append((trainer, kwargs))


class RecordingComputationalCallback(ComputationalCallback):
    """Computational callback; receives the raw predictions dict and returns metrics."""

    def __init__(self):
        self.valid_step_calls = []

    def on_valid_step(self, trainer, step, val_id, predictions_dict, **kwargs):
        self.valid_step_calls.append((trainer, step, val_id, predictions_dict, kwargs))
        return {"dummy_metric": 1.0}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
@pytest.fixture
def module():
    return DummyModule()


@pytest.fixture
def training_protocol(module):
    return DummyTrainingProtocol(module)


@pytest.fixture
def inference_protocol(module):
    return DummyInferenceProtocol(module)


@pytest.fixture
def opt_manager():
    return DummyOptManager()


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
class TestTrainer:
    # ---- Construction -----------------------------------------------------
    def test_init(self, training_protocol, inference_protocol, opt_manager):
        callbacks = [RecordingCallback()]
        trainer = Trainer(
            training_protocol,
            opt_manager,
            inference_protocol=inference_protocol,
            callbacks=callbacks,
        )

        assert trainer.training_protocol is training_protocol
        assert trainer.inference_protocol is inference_protocol
        assert trainer.opt_manager is opt_manager
        assert len(trainer._callbacks) == 1
        assert trainer.train_logs_raw == []
        assert trainer.val_logs_raw == {}
        assert trainer.current_step == 0

    def test_init_without_inference_protocol(self, training_protocol, opt_manager):
        """Inference protocol is optional; validation is skipped when absent."""
        trainer = Trainer(training_protocol, opt_manager)
        assert trainer.inference_protocol is None

    def test_init_rejects_bad_callbacks(self, training_protocol, opt_manager):
        with pytest.raises(TypeError, match="callbacks"):
            Trainer(training_protocol, opt_manager, callbacks=42)

    # ---- Log appenders ----------------------------------------------------
    def test_append_train_log(self, training_protocol, opt_manager):
        trainer = Trainer(training_protocol, opt_manager)
        trainer._append_train_log({"loss": 0.5})
        assert len(trainer.train_logs_raw) == 1
        assert trainer.train_logs_raw[0]["loss"] == 0.5

    def test_append_val_log_new_key(self, training_protocol, opt_manager):
        trainer = Trainer(training_protocol, opt_manager)
        trainer._append_val_log("val1", {"metric": 0.5})
        assert "val1" in trainer.val_logs_raw
        assert trainer.val_logs_raw["val1"][0]["metric"] == 0.5

    def test_append_val_log_existing_key(self, training_protocol, opt_manager):
        trainer = Trainer(training_protocol, opt_manager)
        trainer._append_val_log("val1", {"metric": 0.5})
        trainer._append_val_log("val1", {"metric": 0.8})
        assert len(trainer.val_logs_raw["val1"]) == 2

    # ---- Validation pass --------------------------------------------------
    def test_run_val_on_loader(self, training_protocol, inference_protocol, opt_manager):
        callback = RecordingCallback()
        metric_cb = RecordingComputationalCallback()
        trainer = Trainer(
            training_protocol,
            opt_manager,
            inference_protocol=inference_protocol,
            callbacks=[metric_cb, callback],
        )
        trainer._current_step = 5

        trainer._run_val_on_loader(DummyValLoader(), "test_val")

        # The val log holds the metrics the callbacks computed, tagged with the step.
        assert "test_val" in trainer.val_logs_raw
        assert len(trainer.val_logs_raw["test_val"]) == 1
        log_entry = trainer.val_logs_raw["test_val"][0]
        assert log_entry["dummy_metric"] == 1.0
        assert log_entry["step"] == 5

        # The raw predictions/targets reach the computational callback, one entry per node.
        assert len(metric_cb.valid_step_calls) == 1
        predictions_dict = metric_cb.valid_step_calls[0][3]
        assert len(predictions_dict) == 2
        assert all(set(v) == {"predictions", "targets"} for v in predictions_dict.values())

        # The logging callback is also notified.
        assert len(callback.valid_step_calls) == 1
        assert callback.valid_step_calls[0][1] == 5
        assert callback.valid_step_calls[0][2] == "test_val"

    def test_run_val_on_loader_no_inference_protocol(self, training_protocol, opt_manager):
        """When no inference protocol is set, validation is a no-op."""
        callback = RecordingCallback()
        trainer = Trainer(training_protocol, opt_manager, callbacks=[callback])
        trainer._run_val_on_loader(DummyValLoader(), "test_val")
        assert trainer.val_logs_raw == {}
        assert callback.valid_step_calls == []

    # ---- Log DataFrame conversion ----------------------------------------
    def test_get_train_logs_df_empty(self, training_protocol, opt_manager):
        trainer = Trainer(training_protocol, opt_manager)
        df = trainer.get_train_logs_df()
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_get_train_logs_df_with_data(self, training_protocol, opt_manager):
        trainer = Trainer(training_protocol, opt_manager)
        trainer._append_train_log({"loss": 0.5, "step": 0})
        trainer._append_train_log({"loss": 0.3, "step": 1})

        df = trainer.get_train_logs_df()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "loss" in df.columns
        # `step` becomes the index, so the columns are metrics only.
        assert df.index.name == "step"
        assert "step" not in df.columns
        assert list(df.index) == [0, 1]
        assert list(df["loss"]) == [0.5, 0.3]

    def test_get_val_logs_df_single(self, training_protocol, opt_manager):
        trainer = Trainer(training_protocol, opt_manager)
        trainer._append_val_log("val1", {"metric": 0.5})

        df = trainer.get_val_logs_df("val1")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert df.iloc[0]["metric"] == 0.5

    def test_get_val_logs_df_missing(self, training_protocol, opt_manager):
        trainer = Trainer(training_protocol, opt_manager)
        df = trainer.get_val_logs_df("nonexistent")
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_get_val_logs_df_all(self, training_protocol, opt_manager):
        trainer = Trainer(training_protocol, opt_manager)
        trainer._append_val_log("val1", {"metric": 0.5})
        trainer._append_val_log("val2", {"metric": 0.8})

        result = trainer.get_val_logs_df()
        assert isinstance(result, dict)
        assert set(result.keys()) == {"val1", "val2"}
        assert isinstance(result["val1"], pd.DataFrame)
        assert len(result["val1"]) == 1

    # ---- Training loop ----------------------------------------------------
    @patch("sckitflow.trainer._trainer.tqdm")
    def test_train_calls_callbacks(self, mock_tqdm, training_protocol, opt_manager):
        mock_pbar = MagicMock()
        mock_pbar.__iter__.return_value = range(3)
        mock_tqdm.return_value = mock_pbar

        callback = RecordingCallback()
        trainer = Trainer(training_protocol, opt_manager, callbacks=[callback])
        trainer.train(DummyTrainLoader())

        assert len(callback.train_begin_calls) == 1
        assert len(callback.train_step_calls) == 3
        assert len(callback.train_end_calls) == 1

    @patch("sckitflow.trainer._trainer.tqdm")
    def test_train_with_validation(self, mock_tqdm, training_protocol, inference_protocol, opt_manager):
        mock_tqdm.side_effect = lambda steps: steps

        callback = RecordingCallback()
        trainer = Trainer(
            training_protocol,
            opt_manager,
            inference_protocol=inference_protocol,
            callbacks=[callback],
        )
        # 5 steps -> validate at 2, 4.
        trainer.train(
            DummyTrainLoader(5),
            val_loaders={"val1": DummyValLoader()},
            valid_freq=2,
        )

        assert [call[1] for call in callback.valid_step_calls] == [2, 4]

    @patch("sckitflow.trainer._trainer.tqdm")
    def test_train_without_inference_protocol_skips_validation(self, mock_tqdm, training_protocol, opt_manager):
        """Even with val_loaders, no inference protocol means no validation metrics."""
        mock_tqdm.side_effect = lambda steps: steps

        callback = RecordingCallback()
        trainer = Trainer(training_protocol, opt_manager, callbacks=[callback])
        trainer.train(
            DummyTrainLoader(3),
            val_loaders={"val1": DummyValLoader()},
            valid_freq=1,
        )
        # No metrics from a val run: the log stays empty.
        assert trainer.val_logs_raw == {}
        assert callback.valid_step_calls == []

    @patch("sckitflow.trainer._trainer.tqdm")
    def test_train_continues_from_current_step(self, mock_tqdm, training_protocol, inference_protocol, opt_manager):
        mock_tqdm.side_effect = lambda steps: steps

        callback = RecordingCallback()
        trainer = Trainer(
            training_protocol,
            opt_manager,
            inference_protocol=inference_protocol,
            callbacks=[RecordingComputationalCallback(), callback],
        )
        loader = DummyTrainLoader(3)  # 3 steps per call; two calls -> 6
        val_loaders = {"val1": DummyValLoader()}

        trainer.train(loader, val_loaders=val_loaders, valid_freq=2)
        trainer.train(loader, val_loaders=val_loaders, valid_freq=2)

        assert trainer.current_step == 6
        assert [call[1] for call in callback.train_step_calls] == [1, 2, 3, 4, 5, 6]
        assert list(trainer.get_val_logs_df("val1").index) == [2, 4, 6]

    # ---- Properties -------------------------------------------------------
    def test_properties(self, training_protocol, inference_protocol, opt_manager):
        trainer = Trainer(
            training_protocol,
            opt_manager,
            inference_protocol=inference_protocol,
        )
        assert trainer.training_protocol is training_protocol
        assert trainer.inference_protocol is inference_protocol
        assert trainer.opt_manager is opt_manager
        assert trainer.train_logs_raw == []
        assert trainer.val_logs_raw == {}
