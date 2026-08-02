from unittest.mock import Mock

import lightning.pytorch as pl
import numpy as np
import pytest
import torch
from torchmetrics import Metric

from sckitflow.trainer._callbacks import MetricsCallback


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
class SumMetric(Metric):
    """Accumulates ``pred.sum() - target.sum()`` over every update."""

    def __init__(self):
        super().__init__()
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        self.total += pred.sum() - target.sum()
        self.n += 1

    def compute(self) -> torch.Tensor:
        return self.total / self.n


def _outputs(pred, target):
    return {"predictions": pred, "targets": target}


def _feed(callback, outputs, pl_module, dataloader_idx=0, trainer=None):
    """Drives one validation node through the callback, returning the trainer used."""
    trainer = trainer if trainer is not None else Mock(val_ids=["valA", "valB"])
    callback.on_validation_batch_end(trainer, pl_module, outputs, Mock(), 0, dataloader_idx)
    return trainer


@pytest.fixture
def pl_module():
    module = Mock()
    module.device = torch.device("cpu")
    module.log = Mock()
    return module


# -----------------------------------------------------------------------------
# Construction
# -----------------------------------------------------------------------------
class TestMetricsCallbackInit:
    def test_is_a_lightning_callback(self):
        assert isinstance(MetricsCallback(metrics={}), pl.Callback)

    def test_init_with_instantiated_metrics(self):
        metric = SumMetric()
        callback = MetricsCallback(metrics={"sum": metric})
        assert callback.metrics == {"sum": metric}

    def test_shared_transforms_apply_to_both_sides(self):
        fn = Mock()
        callback = MetricsCallback(metrics={}, transforms=fn)
        assert callback._pred_transforms == [fn]
        assert callback._target_transforms == [fn]

    def test_side_specific_transforms_override_the_shared_one(self):
        shared, pred_only, target_only = Mock(), Mock(), Mock()
        callback = MetricsCallback(
            metrics={},
            transforms=shared,
            pred_transforms=pred_only,
            target_transforms=target_only,
        )
        assert callback._pred_transforms == [pred_only]
        assert callback._target_transforms == [target_only]

    @pytest.mark.parametrize(("given", "expected_len"), [(None, 0), (len, 1)])
    def test_to_list_normalizes_scalars(self, given, expected_len):
        assert len(MetricsCallback._to_list(given)) == expected_len

    def test_to_list_keeps_sequences(self):
        fns = [Mock(), Mock()]
        assert MetricsCallback._to_list(fns) == fns


# -----------------------------------------------------------------------------
# Tensor conversion
# -----------------------------------------------------------------------------
class TestToTensor:
    def test_numpy_is_converted(self):
        out = MetricsCallback._to_tensor(np.ones((2, 3)), torch.device("cpu"))
        assert isinstance(out, torch.Tensor)
        assert out.shape == (2, 3)
        assert out.dtype is torch.float32

    def test_a_tensor_is_passed_through(self):
        tensor = torch.ones(2, 3)
        assert MetricsCallback._to_tensor(tensor, torch.device("cpu")) is tensor

    def test_array_likes_are_converted(self):
        class ArrayLike:
            def __array__(self, dtype=None):
                return np.ones((2, 2))

        out = MetricsCallback._to_tensor(ArrayLike(), torch.device("cpu"))
        assert isinstance(out, torch.Tensor)
        assert out.shape == (2, 2)

    def test_plain_sequences_are_converted(self):
        out = MetricsCallback._to_tensor([[1.0, 2.0]], torch.device("cpu"))
        assert isinstance(out, torch.Tensor)
        assert out.shape == (1, 2)


# -----------------------------------------------------------------------------
# Accumulation and logging
# -----------------------------------------------------------------------------
class TestValidationFlow:
    def test_metrics_accumulate_across_nodes_then_log_once(self, pl_module):
        callback = MetricsCallback(metrics={"sum": SumMetric()})

        trainer = _feed(callback, _outputs(np.ones((2, 2)), np.zeros((2, 2))), pl_module)
        _feed(callback, _outputs(np.full((2, 2), 3.0), np.zeros((2, 2))), pl_module, trainer=trainer)
        callback.on_validation_epoch_end(trainer, pl_module)

        # (4 - 0) then (12 - 0) over two updates -> mean 8.
        pl_module.log.assert_called_once()
        name, value = pl_module.log.call_args.args
        assert name == "valA_sum"
        assert value == pytest.approx(8.0)

    def test_logged_value_is_a_python_scalar(self, pl_module):
        callback = MetricsCallback(metrics={"sum": SumMetric()})
        trainer = _feed(callback, _outputs(np.ones((2, 2)), np.zeros((2, 2))), pl_module)
        callback.on_validation_epoch_end(trainer, pl_module)

        assert isinstance(pl_module.log.call_args.args[1], float)

    def test_batch_size_is_pinned_so_lightning_never_inspects_a_node(self, pl_module):
        callback = MetricsCallback(metrics={"sum": SumMetric()})
        trainer = _feed(callback, _outputs(np.ones((2, 2)), np.zeros((2, 2))), pl_module)
        callback.on_validation_epoch_end(trainer, pl_module)

        assert pl_module.log.call_args.kwargs["batch_size"] == 1

    def test_transforms_are_applied_per_side(self, pl_module):
        callback = MetricsCallback(
            metrics={"sum": SumMetric()},
            pred_transforms=lambda x: x * 2,
        )
        trainer = _feed(callback, _outputs(np.ones((2, 2)), np.zeros((2, 2))), pl_module)
        callback.on_validation_epoch_end(trainer, pl_module)

        # Predictions doubled: 4 * 2 = 8, targets untouched.
        assert pl_module.log.call_args.args[1] == pytest.approx(8.0)

    def test_state_is_reset_between_validation_runs(self, pl_module):
        callback = MetricsCallback(metrics={"sum": SumMetric()})

        trainer = _feed(callback, _outputs(np.ones((2, 2)), np.zeros((2, 2))), pl_module)
        callback.on_validation_epoch_end(trainer, pl_module)

        callback.on_validation_start(trainer, pl_module)
        _feed(callback, _outputs(np.full((2, 2), 5.0), np.zeros((2, 2))), pl_module, trainer=trainer)
        callback.on_validation_epoch_end(trainer, pl_module)

        # The second run reports only its own node: 20, not a running total.
        assert pl_module.log.call_args.args[1] == pytest.approx(20.0)

    @pytest.mark.parametrize("outputs", [None, {}, {"predictions": None, "targets": np.zeros((2, 2))}])
    def test_incomplete_outputs_are_skipped(self, pl_module, outputs):
        callback = MetricsCallback(metrics={"sum": SumMetric()})
        trainer = _feed(callback, outputs, pl_module)
        callback.on_validation_epoch_end(trainer, pl_module)

        pl_module.log.assert_not_called()


class TestPerDataloaderState:
    def test_each_validation_set_gets_its_own_metric_state(self, pl_module):
        """Two validation sets must not pool their accumulated state."""
        callback = MetricsCallback(metrics={"sum": SumMetric()})

        trainer = _feed(callback, _outputs(np.ones((2, 2)), np.zeros((2, 2))), pl_module, dataloader_idx=0)
        _feed(
            callback,
            _outputs(np.full((2, 2), 10.0), np.zeros((2, 2))),
            pl_module,
            dataloader_idx=1,
            trainer=trainer,
        )
        callback.on_validation_epoch_end(trainer, pl_module)

        logged = {call.args[0]: call.args[1] for call in pl_module.log.call_args_list}
        assert logged["valA_sum"] == pytest.approx(4.0)
        assert logged["valB_sum"] == pytest.approx(40.0)

    def test_names_come_from_the_trainers_val_ids(self, pl_module):
        callback = MetricsCallback(metrics={"sum": SumMetric()})
        trainer = Mock(val_ids=["first", "second"])

        _feed(callback, _outputs(np.ones((2, 2)), np.zeros((2, 2))), pl_module, 1, trainer)
        callback.on_validation_epoch_end(trainer, pl_module)

        assert pl_module.log.call_args.args[0] == "second_sum"

    def test_positional_fallback_when_no_val_ids_are_available(self, pl_module):
        callback = MetricsCallback(metrics={"sum": SumMetric()})
        trainer = Mock(spec=[])  # a plain pl.Trainer has no `val_ids`

        _feed(callback, _outputs(np.ones((2, 2)), np.zeros((2, 2))), pl_module, 2, trainer)
        callback.on_validation_epoch_end(trainer, pl_module)

        assert pl_module.log.call_args.args[0] == "val2_sum"

    def test_every_dataloader_gets_its_own_copy(self, pl_module):
        metric = SumMetric()
        callback = MetricsCallback(metrics={"sum": metric})

        trainer = _feed(callback, _outputs(np.ones((2, 2)), np.zeros((2, 2))), pl_module, 0)
        _feed(callback, _outputs(np.ones((2, 2)), np.zeros((2, 2))), pl_module, 1, trainer)

        assert callback._per_loader[0]["sum"] is not metric
        assert callback._per_loader[1]["sum"] is not metric
        assert callback._per_loader[0]["sum"] is not callback._per_loader[1]["sum"]

    def test_the_callers_metric_is_never_mutated(self, pl_module):
        """The template must stay pristine, or later dataloaders inherit its state."""
        metric = SumMetric()
        callback = MetricsCallback(metrics={"sum": metric})

        _feed(callback, _outputs(np.full((2, 2), 7.0), np.zeros((2, 2))), pl_module)

        assert metric.total == pytest.approx(0.0)
        assert metric.n == pytest.approx(0.0)

    def test_a_later_dataloader_does_not_inherit_an_earlier_ones_state(self, pl_module):
        """Regression: cloning from an already-updated metric pooled the two sets."""
        callback = MetricsCallback(metrics={"sum": SumMetric()})

        trainer = _feed(callback, _outputs(np.ones((2, 2)), np.zeros((2, 2))), pl_module, 0)
        _feed(callback, _outputs(np.full((2, 2), 10.0), np.zeros((2, 2))), pl_module, 1, trainer)

        # The second set saw exactly one node, so `n` must be 1 and not 2.
        assert callback._per_loader[1]["sum"].n == pytest.approx(1.0)
