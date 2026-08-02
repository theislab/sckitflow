import lightning.pytorch as pl
import pandas as pd
import pytest
import torch

from sckitflow.trainer._logger import DataFrameLogger
from sckitflow.trainer._trainer import Trainer


# -----------------------------------------------------------------------------
# Minimal stand-ins for the method and the samplers
# -----------------------------------------------------------------------------
class Node:
    """Stand-in for a `MatchedDistributions` node: a custom, non-tensor batch."""

    def __init__(self, idx: int) -> None:
        self.idx = idx
        self.n_target_obs = 4
        self.x = torch.randn(4, 3)


class NodeStream:
    """Unbounded per-node stream, like `FTrainSampler`."""

    def __iter__(self):
        idx = 0
        while True:
            yield Node(idx)
            idx += 1


class FiniteNodeStream:
    """Finite, pre-registered nodes, like `FValidationSampler`."""

    def __init__(self, n: int) -> None:
        self._data = [Node(i) for i in range(n)]

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self):
        yield from self._data


class CountingMethod(pl.LightningModule):
    """Records every training step so the node-step accounting can be asserted."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)
        self.train_nodes: list[int] = []
        self.val_calls: list[tuple[int, int]] = []

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        return batch

    def training_step(self, node: Node, batch_idx: int) -> torch.Tensor:
        self.train_nodes.append(node.idx)
        loss = self.linear(node.x).pow(2).mean()
        self.log_dict({"loss": loss}, on_step=True, on_epoch=False, batch_size=node.n_target_obs)
        return loss

    def validation_step(self, node: Node, batch_idx: int, dataloader_idx: int = 0):
        self.val_calls.append((self.global_step, dataloader_idx))
        # `batch_size` is explicit because Lightning cannot infer one from a node.
        self.log(
            f"val{dataloader_idx}_score",
            torch.tensor(1.0),
            on_step=False,
            on_epoch=True,
            batch_size=node.n_target_obs,
        )

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


def _trainer(**kwargs) -> Trainer:
    """A Trainer with the noise turned off, so tests stay quiet and CPU-bound."""
    defaults = {
        "accelerator": "cpu",
        "enable_progress_bar": False,
        "enable_model_summary": False,
    }
    return Trainer(**{**defaults, **kwargs})


# -----------------------------------------------------------------------------
# Construction and defaults
# -----------------------------------------------------------------------------
class TestTrainerDefaults:
    def test_is_a_lightning_trainer(self):
        assert isinstance(_trainer(), pl.Trainer)

    def test_steps_map_onto_lightning_settings(self):
        trainer = _trainer(n_train_steps=7, valid_freq=3)
        assert trainer.max_steps == 7
        assert trainer.val_check_interval == 3

    def test_epoch_based_validation_is_disabled(self):
        """The training stream never ends, so validation must be driven by steps alone."""
        assert _trainer().check_val_every_n_epoch is None

    def test_checkpointing_and_sanity_checks_are_off_by_default(self):
        trainer = _trainer()
        assert trainer.checkpoint_callbacks == []
        assert trainer.num_sanity_val_steps == 0

    def test_checkpointing_can_be_turned_on(self):
        trainer = _trainer(enable_checkpointing=True)
        assert len(trainer.checkpoint_callbacks) == 1

    def test_every_step_reaches_the_logger_by_default(self):
        assert _trainer().log_every_n_steps == 1

    def test_a_dataframe_logger_is_installed_by_default(self):
        assert isinstance(_trainer().logger, DataFrameLogger)

    def test_val_ids_are_exposed_as_a_copy(self):
        trainer = _trainer(val_ids=["valA"])
        trainer.val_ids.append("mutated")
        assert trainer.val_ids == ["valA"]

    def test_no_val_ids_means_an_empty_list(self):
        assert _trainer().val_ids == []


class TestLoggerComposition:
    def test_a_user_logger_is_kept_alongside_the_dataframe_logger(self):
        user_logger = pl.loggers.CSVLogger(save_dir="/tmp/sckitflow-test-logs")
        trainer = _trainer(logger=user_logger)

        assert user_logger in trainer.loggers
        assert any(isinstance(lg, DataFrameLogger) for lg in trainer.loggers)

    def test_a_list_of_user_loggers_is_extended(self):
        user_logger = pl.loggers.CSVLogger(save_dir="/tmp/sckitflow-test-logs")
        trainer = _trainer(logger=[user_logger])

        assert user_logger in trainer.loggers
        assert any(isinstance(lg, DataFrameLogger) for lg in trainer.loggers)

    def test_logging_can_be_opted_out_of_entirely(self):
        trainer = _trainer(logger=False)
        assert trainer.loggers == []
        # The accessors stay usable, they are just empty.
        assert trainer.get_train_logs_df().empty


class TestLogAccessors:
    """The DataFrame accessors delegate to the installed `DataFrameLogger`."""

    def test_empty_before_training(self):
        trainer = _trainer(val_ids=["valA"])
        assert trainer.get_train_logs_df().empty
        assert trainer.train_logs_raw == []
        assert trainer.val_logs_raw == {"valA": []}

    def test_unknown_val_id_yields_an_empty_frame(self):
        assert _trainer(val_ids=["valA"]).get_val_logs_df("nope").empty


# -----------------------------------------------------------------------------
# Node-step accounting -- the core semantics of this trainer
# -----------------------------------------------------------------------------
class TestNodeSteps:
    def test_one_node_is_one_optimizer_step(self):
        """`n_train_steps` counts nodes, and each node drives exactly one step."""
        method = CountingMethod()
        trainer = _trainer(n_train_steps=5, valid_freq=100)

        trainer.fit(method, train_dataloaders=NodeStream())

        assert method.train_nodes == [0, 1, 2, 3, 4]
        # `global_step` counts optimizer steps; equality proves one step per node.
        assert trainer.global_step == 5
        assert trainer.current_step == 5

    def test_training_stops_at_n_train_steps_despite_an_endless_stream(self):
        method = CountingMethod()
        trainer = _trainer(n_train_steps=3, valid_freq=100)

        trainer.fit(method, train_dataloaders=NodeStream())

        assert len(method.train_nodes) == 3

    def test_train_logs_record_one_row_per_node_step(self):
        method = CountingMethod()
        trainer = _trainer(n_train_steps=4, valid_freq=100)

        trainer.fit(method, train_dataloaders=NodeStream())
        df = trainer.get_train_logs_df()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 4
        assert df.index.name == "step"
        assert "loss" in df.columns
        assert "step" not in df.columns

    def test_validation_cadence_is_measured_in_node_steps(self):
        method = CountingMethod()
        trainer = _trainer(n_train_steps=6, valid_freq=2)

        trainer.fit(method, train_dataloaders=NodeStream(), val_dataloaders=FiniteNodeStream(1))

        # Validation at every second node-step: three runs over six steps.
        assert len(method.val_calls) == 3


class TestValidationRouting:
    def test_metrics_are_split_per_validation_set(self):
        method = CountingMethod()
        trainer = _trainer(n_train_steps=2, valid_freq=2, val_ids=["val0", "val1"])

        trainer.fit(
            method,
            train_dataloaders=NodeStream(),
            val_dataloaders=[FiniteNodeStream(2), FiniteNodeStream(3)],
        )

        per_id = trainer.get_val_logs_df()
        assert set(per_id) == {"val0", "val1"}
        # Logging from inside `validation_step` with more than one val dataloader makes
        # Lightning suffix the name with `/dataloader_idx_N` (pass
        # `add_dataloader_idx=False` to suppress it). Routing keys off the prefix, so the
        # metric still lands in the right frame either way.
        for val_id, frame in per_id.items():
            assert frame.columns.tolist()
            assert all(col.startswith(f"{val_id}_") for col in frame.columns)

    def test_every_validation_dataloader_is_visited(self):
        method = CountingMethod()
        trainer = _trainer(n_train_steps=2, valid_freq=2, val_ids=["val0", "val1"])

        trainer.fit(
            method,
            train_dataloaders=NodeStream(),
            val_dataloaders=[FiniteNodeStream(2), FiniteNodeStream(3)],
        )

        visited = [idx for _, idx in method.val_calls]
        assert visited.count(0) == 2  # two nodes in the first set
        assert visited.count(1) == 3  # three nodes in the second


class TestExtraTrainerKwargs:
    def test_lightning_kwargs_pass_through(self):
        """Gradient clipping and the like come free with `pl.Trainer`."""
        trainer = _trainer(gradient_clip_val=0.5, accumulate_grad_batches=2)
        assert trainer.gradient_clip_val == 0.5
        assert trainer.accumulate_grad_batches == 2

    def test_early_stopping_callback_is_accepted(self):
        callback = pl.callbacks.EarlyStopping(monitor="val0_score")
        trainer = _trainer(callbacks=[callback])
        assert callback in trainer.callbacks


@pytest.mark.parametrize("n_train_steps", [1, 3, 8])
def test_global_step_always_equals_n_train_steps(n_train_steps):
    """Node-steps are the only counter that matters, at any run length."""
    method = CountingMethod()
    trainer = _trainer(n_train_steps=n_train_steps, valid_freq=1000)

    trainer.fit(method, train_dataloaders=NodeStream())

    assert trainer.global_step == n_train_steps
    assert len(method.train_nodes) == n_train_steps
