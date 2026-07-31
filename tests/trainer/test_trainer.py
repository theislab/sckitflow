# tests/trainer/test_trainer.py
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd

from sckitflow.methods._methods import BaseMethod
from sckitflow.methods._opt import BaseOptManager
from sckitflow.trainer._callbacks import ComputationalCallback, LoggingCallback
from sckitflow.trainer._trainer import Trainer


# -----------------------------------------------------------------------------
# Dummy classes for testing
# -----------------------------------------------------------------------------
class DummyMethod(BaseMethod):
    _module_cls = None

    def __init__(self, *args, **kwargs):
        self._backend = kwargs.get("backend", "torch")
        self._train_mode = True

    def set_train_mode(self, mode):
        self._train_mode = mode

    def train_step(self, node, *args, **kwargs):
        return 0.5, {"loss": 0.5, "accuracy": 0.8}

    def predict(self, node, *args, **kwargs):
        return np.random.randn(10, 5)


class DummyOptManager(BaseOptManager):
    def step(self, loss):
        pass


class DummySampler:
    def sample(self):
        return [Mock(), Mock()]


class DummyValidationSampler:
    def __iter__(self):
        return iter([Mock(), Mock()])


class RecordingCallback(LoggingCallback):
    # Logging: `TrainingCallbacks` only forwards `on_train_step` to logging callbacks.
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
    # Computational callbacks are the ones handed the raw predictions dict.
    def __init__(self):
        self.valid_step_calls = []

    def on_valid_step(self, trainer, step, val_id, predictions_dict, **kwargs):
        self.valid_step_calls.append((trainer, step, val_id, predictions_dict, kwargs))
        return {"dummy_metric": 1.0}


# -----------------------------------------------------------------------------
# Tests for Trainer
# -----------------------------------------------------------------------------
class TestTrainer:
    def test_init(self):
        method = DummyMethod()
        opt_manager = DummyOptManager()
        callbacks = [RecordingCallback()]

        trainer = Trainer(method, opt_manager, callbacks)

        assert trainer._method is method
        assert trainer._opt_manager is opt_manager
        assert len(trainer._callbacks) == 1
        assert trainer._train_logs == []
        assert trainer._val_logs == {}

    def test_append_train_log(self):
        trainer = Trainer(DummyMethod(), DummyOptManager())
        trainer._append_train_log({"loss": 0.5})
        assert len(trainer._train_logs) == 1
        assert trainer._train_logs[0]["loss"] == 0.5

    def test_append_val_log_new_key(self):
        trainer = Trainer(DummyMethod(), DummyOptManager())
        trainer._append_val_log("val1", {"metric": 0.5})
        assert "val1" in trainer._val_logs
        assert len(trainer._val_logs["val1"]) == 1
        assert trainer._val_logs["val1"][0]["metric"] == 0.5

    def test_append_val_log_existing_key(self):
        trainer = Trainer(DummyMethod(), DummyOptManager())
        trainer._append_val_log("val1", {"metric": 0.5})
        trainer._append_val_log("val1", {"metric": 0.8})
        assert len(trainer._val_logs["val1"]) == 2

    def test_run_val_on_sampler(self):
        method = DummyMethod()
        opt_manager = DummyOptManager()
        callback = RecordingCallback()
        metric_cb = RecordingComputationalCallback()
        trainer = Trainer(method, opt_manager, [metric_cb, callback])

        sampler = DummyValidationSampler()
        trainer._run_val_on_sampler(sampler, "test_val", step=5)

        # The val log holds the metrics the callbacks computed, tagged with the step.
        assert "test_val" in trainer._val_logs
        assert len(trainer._val_logs["test_val"]) == 1
        log_entry = trainer._val_logs["test_val"][0]
        assert log_entry["dummy_metric"] == 1.0
        assert log_entry["step"] == 5

        # The raw predictions/targets reach the computational callback, one entry per node.
        assert len(metric_cb.valid_step_calls) == 1
        predictions_dict = metric_cb.valid_step_calls[0][3]
        assert len(predictions_dict) == 2
        assert all(set(v) == {"predictions", "targets"} for v in predictions_dict.values())

        # Check callback was called
        assert len(callback.valid_step_calls) == 1
        assert callback.valid_step_calls[0][1] == 5
        assert callback.valid_step_calls[0][2] == "test_val"

    def test_get_train_logs_df_empty(self):
        trainer = Trainer(DummyMethod(), DummyOptManager())
        df = trainer.get_train_logs_df()
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_get_train_logs_df_with_data(self):
        trainer = Trainer(DummyMethod(), DummyOptManager())
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

    def test_get_val_logs_df_single(self):
        trainer = Trainer(DummyMethod(), DummyOptManager())
        trainer._append_val_log("val1", {"metric": 0.5})

        df = trainer.get_val_logs_df("val1")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert df.iloc[0]["metric"] == 0.5

    def test_get_val_logs_df_missing(self):
        trainer = Trainer(DummyMethod(), DummyOptManager())
        df = trainer.get_val_logs_df("nonexistent")
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_get_val_logs_df_all(self):
        trainer = Trainer(DummyMethod(), DummyOptManager())
        trainer._append_val_log("val1", {"metric": 0.5})
        trainer._append_val_log("val2", {"metric": 0.8})

        result = trainer.get_val_logs_df()
        assert isinstance(result, dict)
        assert "val1" in result
        assert "val2" in result
        assert isinstance(result["val1"], pd.DataFrame)
        assert len(result["val1"]) == 1

    @patch("sckitflow.trainer._trainer.tqdm")
    def test_train_calls_callbacks(self, mock_tqdm):
        mock_pbar = MagicMock()
        mock_pbar.__iter__.return_value = range(3)
        mock_tqdm.return_value = mock_pbar

        method = DummyMethod()
        opt_manager = DummyOptManager()
        callback = RecordingCallback()

        trainer = Trainer(method, opt_manager, [callback])
        train_sampler = DummySampler()

        trainer.train(train_sampler, n_train_steps=3)

        assert len(callback.train_begin_calls) == 1
        assert len(callback.train_step_calls) == 3
        assert len(callback.train_end_calls) == 1

    @patch("sckitflow.trainer._trainer.tqdm")
    def test_train_with_validation(self, mock_tqdm):
        mock_tqdm.side_effect = lambda steps: steps

        method = DummyMethod()
        opt_manager = DummyOptManager()
        callback = RecordingCallback()

        trainer = Trainer(method, opt_manager, [callback])
        train_sampler = DummySampler()
        val_samplers = {"val1": DummyValidationSampler()}

        trainer.train(train_sampler, val_samplers_dict=val_samplers, n_train_steps=5, valid_freq=2)

        # Validation frequency is based on cumulative completed training steps.
        valid_calls = callback.valid_step_calls
        assert [call[1] for call in valid_calls] == [2, 4]

    @patch("sckitflow.trainer._trainer.tqdm")
    def test_train_continues_from_current_step(self, mock_tqdm):
        mock_tqdm.side_effect = lambda steps: steps

        callback = RecordingCallback()
        trainer = Trainer(DummyMethod(), DummyOptManager(), [RecordingComputationalCallback(), callback])
        train_sampler = DummySampler()
        val_samplers = {"val1": DummyValidationSampler()}

        trainer.train(train_sampler, val_samplers_dict=val_samplers, n_train_steps=3, valid_freq=2)
        trainer.train(train_sampler, val_samplers_dict=val_samplers, n_train_steps=3, valid_freq=2)

        assert trainer.current_step == 6
        assert [call[1] for call in callback.train_step_calls] == [1, 2, 3, 4, 5, 6]
        assert list(trainer.get_val_logs_df("val1").index) == [2, 4, 6]

    def test_properties(self):
        method = DummyMethod()
        opt_manager = DummyOptManager()
        trainer = Trainer(method, opt_manager)

        assert trainer.opt_manager is opt_manager
        assert trainer.train_logs_raw == []
        assert trainer.val_logs_raw == {}
