# tests/trainer/test_trainer.py
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from sc_flow.methods._methods import BaseMethod
from sc_flow.methods._opt import BaseOptManager
from sc_flow.trainer._trainer import Trainer


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

    def train_step(self, node):
        return 0.5, {"loss": 0.5, "accuracy": 0.8}

    def predict(self, node):
        return np.random.randn(10, 5)

    def extract_state_data(self, state_data):
        return state_data


class DummyJointMethod(DummyMethod):
    @property
    def is_joint(self):
        return True

    def train_step_joint(self, nodes, *args, **kwargs):
        return 0.3, {"loss": 0.3}


class DummyOptManager(BaseOptManager):
    def step(self, loss):
        pass


class DummySampler:
    def sample(self):
        return [Mock(), Mock()]


class DummyValidationSampler:
    def __iter__(self):
        return iter([Mock(), Mock()])


class RecordingCallback:
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
        trainer = Trainer(method, opt_manager, [callback])

        sampler = DummyValidationSampler()
        trainer._run_val_on_sampler(sampler, "test_val", step=5)

        # Check that val log was appended
        assert "test_val" in trainer._val_logs
        assert len(trainer._val_logs["test_val"]) == 1
        assert "predictions" in trainer._val_logs["test_val"][0]
        assert trainer._val_logs["test_val"][0]["step"] == 5

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
        assert "step" in df.columns

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

    @patch("sc_flow.trainer._trainer.tqdm")
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

    @patch("sc_flow.trainer._trainer.tqdm")
    def test_train_with_validation(self, mock_tqdm):
        mock_pbar = MagicMock()
        mock_pbar.__iter__.return_value = range(5)
        mock_tqdm.return_value = mock_pbar

        method = DummyMethod()
        opt_manager = DummyOptManager()
        callback = RecordingCallback()

        trainer = Trainer(method, opt_manager, [callback])
        train_sampler = DummySampler()
        val_samplers = {"val1": DummyValidationSampler()}

        trainer.train(train_sampler, val_samplers, n_train_steps=5, valid_freq=2)

        # Should have validation calls at steps 1 and 3 (0-indexed)
        valid_calls = callback.valid_step_calls
        assert len(valid_calls) >= 2

    def test_properties(self):
        method = DummyMethod()
        opt_manager = DummyOptManager()
        trainer = Trainer(method, opt_manager)

        assert trainer.opt_manager is opt_manager
        assert trainer.train_logs_raw == []
        assert trainer.val_logs_raw == {}


class TestTrainerExecutorSelection:
    def test_independent_executor_calls_per_node(self):
        method = DummyMethod()
        method.train_step = Mock(return_value=(0.5, {"loss": 0.5}))
        trainer = Trainer(method, DummyOptManager())
        nodes = [Mock(), Mock(), Mock()]

        results = trainer._independent_train_executor(method, nodes)

        assert len(results) == 3
        assert method.train_step.call_count == 3

    def test_join_executor_calls_once(self):
        method = DummyJointMethod()
        method.train_step_joint = Mock(return_value=(0.3, {"loss": 0.3}))
        trainer = Trainer(method, DummyOptManager())
        nodes = [Mock(), Mock(), Mock()]

        results = trainer._join_train_executor(method, nodes)

        assert len(results) == 1
        method.train_step_joint.assert_called_once()

    @patch("sc_flow.trainer._trainer.tqdm")
    def test_is_joint_false_selects_independent(self, mock_tqdm):
        mock_pbar = MagicMock()
        mock_pbar.__iter__.return_value = range(2)
        mock_tqdm.return_value = mock_pbar

        method = DummyMethod()
        method.train_step = Mock(return_value=(0.5, {"loss": 0.5}))
        trainer = Trainer(method, DummyOptManager())
        sampler = DummySampler()

        trainer.train(sampler, n_train_steps=2)

        # DummySampler returns 2 nodes per sample call, 2 steps = 4 train_step calls
        assert method.train_step.call_count == 4

    @patch("sc_flow.trainer._trainer.tqdm")
    def test_is_joint_true_selects_join(self, mock_tqdm):
        mock_pbar = MagicMock()
        mock_pbar.__iter__.return_value = range(2)
        mock_tqdm.return_value = mock_pbar

        method = DummyJointMethod()
        method.train_step_joint = Mock(return_value=(0.3, {"loss": 0.3}))
        trainer = Trainer(method, DummyOptManager())
        sampler = DummySampler()

        trainer.train(sampler, n_train_steps=2)

        # Joint: one call per step = 2 calls total
        assert method.train_step_joint.call_count == 2

    def test_base_method_is_joint_false(self):
        method = DummyMethod()
        assert method.is_joint is False

    def test_base_method_train_step_joint_raises(self):
        method = DummyMethod()
        with pytest.raises(NotImplementedError):
            method.train_step_joint()
