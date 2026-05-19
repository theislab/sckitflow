from unittest.mock import MagicMock, Mock, patch

import numpy as np

# import torch
from sc_flow.trainer._callbacks import (
    BaseCallback,
    ComputationalCallback,
    LoggingCallback,
    MetricsCallback,
    TrainingCallbacks,
    WandBLogger,
)


# -----------------------------------------------------------------------------
# Simple mock metrics for testing (no torchmetrics dependency)
# -----------------------------------------------------------------------------
class MockMetric:
    def __init__(self):
        self._value = 0.0
        self._count = 0

    def update(self, pred, target):
        if hasattr(pred, "mean"):
            self._value += ((pred - target) ** 2).mean().item()
        else:
            self._value += np.mean((pred - target) ** 2)
        self._count += 1

    def compute(self):
        return self._value / self._count if self._count > 0 else 0.0

    def reset(self):
        self._value = 0.0
        self._count = 0

    def to(self, device):
        return self


class MockMMD:
    def __init__(self, gammas=None):
        self.gammas = gammas or [1.0, 0.5]
        self._value = 0.0
        self._count = 0

    def update(self, pred, target):
        self._value += 0.1
        self._count += 1

    def compute(self):
        return self._value / self._count if self._count > 0 else 0.0

    def reset(self):
        self._value = 0.0
        self._count = 0

    def to(self, device):
        return self


# -----------------------------------------------------------------------------
# Test base classes
# -----------------------------------------------------------------------------
class TestBaseCallback:
    def test_base_callback_methods_do_nothing(self):
        cb = BaseCallback()
        trainer = Mock()

        cb.on_train_begin(trainer)
        cb.on_train_step(trainer, 0, {})
        cb.on_valid_step(trainer, 0, "val", {})
        cb.on_train_end(trainer)


class TestComputationalCallback:
    def test_computational_callback_type(self):
        cb = ComputationalCallback()
        assert cb.callback_type == "computational"


class TestLoggingCallback:
    def test_logging_callback_type(self):
        cb = LoggingCallback()
        assert cb.callback_type == "logging"


# -----------------------------------------------------------------------------
# Tests for MetricsCallback
# -----------------------------------------------------------------------------
class TestMetricsCallback:
    def test_init_with_instantiated_metrics(self):
        cb = MetricsCallback(metrics={"mse": MockMetric(), "mmd": MockMMD(gammas=[1.0, 0.5])}, backend="torch")
        assert len(cb._metrics) == 2
        assert isinstance(cb._metrics["mse"], MockMetric)
        assert isinstance(cb._metrics["mmd"], MockMMD)
        assert cb._backend == "torch"

    def test_init_with_transforms(self):
        def transform1(x):
            return x * 2

        def transform2(x):
            return x + 1

        cb = MetricsCallback(
            metrics={"mse": MockMetric()},
            backend="torch",
            transforms=transform1,
            pred_transforms=[transform2],
            target_transforms=[transform2],
        )
        # pred_transforms replaces transforms, so only transform2 (1 item)
        assert len(cb._pred_transforms) == 1
        assert len(cb._target_transforms) == 1
        assert cb._pred_transforms[0] is transform2
        assert cb._target_transforms[0] is transform2

    def test_to_list_none(self):
        cb = MetricsCallback(metrics={"mse": MockMetric()}, backend="torch")
        result = cb._to_list(None)
        assert result == []

    def test_to_list_callable(self):
        cb = MetricsCallback(metrics={"mse": MockMetric()}, backend="torch")

        def func(x):
            return x

        result = cb._to_list(func)
        assert len(result) == 1
        assert result[0] is func

    def test_to_list_sequence(self):
        cb = MetricsCallback(metrics={"mse": MockMetric()}, backend="torch")

        def func1(x):
            return x

        def func2(x):
            return x

        result = cb._to_list([func1, func2])
        assert len(result) == 2

    def test_reset_metrics(self):
        cb = MetricsCallback(metrics={"mse": MockMetric()}, backend="torch")
        cb._reset_metrics()
        assert True

    def test_apply_transforms(self):
        def transform(x):
            return x + 1

        cb = MetricsCallback(metrics={"mse": MockMetric()}, backend="torch")
        result = cb._apply_transforms(5, [transform])
        assert result == 6

    # def test_to_tensor_numpy_torch_backend(self):
    #     cb = MetricsCallback(metrics={"mse": MockMetric()}, backend="torch", device="cpu")
    #     data = np.array([1.0, 2.0, 3.0])
    #     tensor = cb._to_tensor(data)
    #     assert isinstance(tensor, torch.Tensor)
    #     assert tensor.shape == (3,)

    # def test_to_tensor_torch_tensor_torch_backend(self):
    #     cb = MetricsCallback(metrics={"mse": MockMetric()}, backend="torch", device="cpu")
    #     data = torch.tensor([1.0, 2.0, 3.0])
    #     tensor = cb._to_tensor(data)
    #     assert tensor is data

    def test_to_tensor_jax_backend_returns_numpy(self):
        cb = MetricsCallback(metrics={"mse": MockMetric()}, backend="jax")
        data = np.array([1.0, 2.0, 3.0])
        result = cb._to_tensor(data)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, data)

    def test_on_valid_step_torch_backend(self):
        cb = MetricsCallback(metrics={"mse": MockMetric()}, backend="torch", device="cpu")

        trainer = Mock()
        predictions_dict = {
            "pert1": {"predictions": np.random.randn(10, 5), "targets": np.random.randn(10, 5)},
            "pert2": {"predictions": np.random.randn(10, 5), "targets": np.random.randn(10, 5)},
        }

        result = cb.on_valid_step(trainer, step=0, val_id="test", predictions_dict=predictions_dict)

        # Metrics are aggregated across all perturbations, so key is "test_mse"
        assert "test_mse" in result
        assert isinstance(result["test_mse"], float)

    def test_on_valid_step_with_transforms(self):
        def add_one(x):
            return x + 1

        cb = MetricsCallback(metrics={"mean": MockMetric()}, backend="torch", transforms=add_one, device="cpu")

        trainer = Mock()
        predictions_dict = {"pert1": {"predictions": np.array([[1.0], [2.0]]), "targets": np.array([[1.0], [2.0]])}}

        result = cb.on_valid_step(trainer, step=0, val_id="test", predictions_dict=predictions_dict)
        # After transform, values become 2 and 3, metric key is "test_mean"
        assert "test_mean" in result
        assert isinstance(result["test_mean"], float)


# -----------------------------------------------------------------------------
# Tests for WandBLogger (patch the lazy import)
# -----------------------------------------------------------------------------
class TestWandBLogger:
    def test_on_train_begin(self):
        mock_wandb = MagicMock()
        mock_wandb.run = MagicMock()
        mock_wandb.run.name = "test_run"

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            logger = WandBLogger(project_name="test_project", log_dir="./logs", config={"lr": 0.001})
            trainer = Mock()
            logger.on_train_begin(trainer)

            mock_wandb.login.assert_called_once()
            mock_wandb.init.assert_called_once()

    def test_on_train_step(self):
        mock_wandb = MagicMock()

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            logger = WandBLogger(project_name="test_project", log_dir="./logs", config={})
            trainer = Mock()
            logs = {"loss": 0.5, "accuracy": 0.8}
            logger.on_train_step(trainer, step=10, logs=logs)

            mock_wandb.log.assert_called_once_with({"train_step": 10, **logs})

    def test_on_valid_step(self):
        mock_wandb = MagicMock()

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            logger = WandBLogger(project_name="test_project", log_dir="./logs", config={})
            trainer = Mock()
            metrics = {"mse": 0.25, "mae": 0.4}
            logger.on_valid_step(trainer, step=10, val_id="val1", predictions_dict=metrics)

            mock_wandb.log.assert_called_once_with({"val_val1_mse": 0.25, "val_val1_mae": 0.4})

    def test_on_train_end(self):
        mock_wandb = MagicMock()

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            logger = WandBLogger(project_name="test_project", log_dir="./logs", config={})
            trainer = Mock()
            logger.on_train_end(trainer)

            mock_wandb.finish.assert_called_once()


# -----------------------------------------------------------------------------
# Tests for TrainingCallbacks
# -----------------------------------------------------------------------------
class TestTrainingCallbacks:
    def test_init(self):
        cb1 = Mock(spec=BaseCallback)
        cb1.callback_type = "logging"
        cb2 = Mock(spec=BaseCallback)
        cb2.callback_type = "logging"

        training_cb = TrainingCallbacks([cb1, cb2])

        assert len(training_cb.callbacks) == 2
        assert len(training_cb._computational) == 0
        assert len(training_cb._logging) == 2

    def test_on_train_begin(self):
        cb1 = Mock(spec=BaseCallback)
        cb2 = Mock(spec=BaseCallback)
        training_cb = TrainingCallbacks([cb1, cb2])

        trainer = Mock()
        training_cb.on_train_begin(trainer, extra="value")

        cb1.on_train_begin.assert_called_once_with(trainer, extra="value")
        cb2.on_train_begin.assert_called_once_with(trainer, extra="value")

    def test_on_train_step(self):
        computational_cb = ComputationalCallback()
        logging_cb = Mock(spec=LoggingCallback)
        logging_cb.callback_type = "logging"

        training_cb = TrainingCallbacks([computational_cb, logging_cb])

        trainer = Mock()
        logs = {"loss": 0.5}
        training_cb.on_train_step(trainer, step=5, logs=logs)

        logging_cb.on_train_step.assert_called_once_with(trainer, 5, logs)

    def test_on_valid_step(self):
        class DummyComputational(ComputationalCallback):
            def on_valid_step(self, trainer, step, val_id, predictions_dict, **kwargs):
                return {"computed_metric": 0.99}

        computational_cb = DummyComputational()
        logging_cb = Mock(spec=LoggingCallback)
        logging_cb.callback_type = "logging"

        training_cb = TrainingCallbacks([computational_cb, logging_cb])

        trainer = Mock()
        predictions = {"test": {"predictions": np.array([1]), "targets": np.array([1])}}

        training_cb.on_valid_step(trainer, step=0, val_id="val", predictions_dict=predictions)

        logging_cb.on_valid_step.assert_called_once()
        args = logging_cb.on_valid_step.call_args[0]
        assert args[3] == {"computed_metric": 0.99}

    def test_on_train_end(self):
        cb1 = Mock(spec=BaseCallback)
        cb2 = Mock(spec=BaseCallback)
        training_cb = TrainingCallbacks([cb1, cb2])

        trainer = Mock()
        training_cb.on_train_end(trainer)

        cb1.on_train_end.assert_called_once_with(trainer)
        cb2.on_train_end.assert_called_once_with(trainer)
