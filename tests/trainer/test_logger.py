import pandas as pd
import torch

from sckitflow.trainer._logger import DataFrameLogger


class TestRouting:
    """A metric belongs to a validation set when its name is prefixed with that id."""

    def test_metrics_without_a_val_prefix_are_training_data(self):
        logger = DataFrameLogger(val_ids=["valA"])
        logger.log_metrics({"loss": 0.5}, step=0)

        assert logger.train_logs_raw == [{"loss": 0.5, "step": 0}]
        assert logger.val_logs_raw == {"valA": []}

    def test_prefixed_metrics_are_routed_to_their_validation_set(self):
        logger = DataFrameLogger(val_ids=["valA", "valB"])
        logger.log_metrics({"valA_mmd": 0.1}, step=3)
        logger.log_metrics({"valB_mmd": 0.2}, step=3)

        assert logger.val_logs_raw["valA"] == [{"valA_mmd": 0.1, "step": 3}]
        assert logger.val_logs_raw["valB"] == [{"valB_mmd": 0.2, "step": 3}]
        assert logger.train_logs_raw == []

    def test_one_call_can_be_split_across_destinations(self):
        """Lightning batches metrics, so a single flush may mix train and val names."""
        logger = DataFrameLogger(val_ids=["valA"])
        logger.log_metrics({"loss": 0.5, "valA_mmd": 0.1}, step=7)

        assert logger.train_logs_raw == [{"loss": 0.5, "step": 7}]
        assert logger.val_logs_raw["valA"] == [{"valA_mmd": 0.1, "step": 7}]

    def test_no_val_ids_means_everything_is_training_data(self):
        logger = DataFrameLogger()
        logger.log_metrics({"valA_mmd": 0.1}, step=1)

        assert logger.train_logs_raw == [{"valA_mmd": 0.1, "step": 1}]
        assert logger.val_logs_raw == {}

    def test_a_val_id_is_not_matched_as_a_bare_substring(self):
        """`"valAB_x"` belongs to `valAB`, not to `valA`."""
        logger = DataFrameLogger(val_ids=["valA"])
        logger.log_metrics({"valAB": 1.0}, step=0)

        assert logger.train_logs_raw == [{"valAB": 1.0, "step": 0}]
        assert logger.val_logs_raw["valA"] == []


class TestRowContents:
    def test_tensor_values_are_unwrapped(self):
        logger = DataFrameLogger()
        logger.log_metrics({"loss": torch.tensor(0.25)}, step=torch.tensor(2))

        assert logger.train_logs_raw == [{"loss": 0.25, "step": 2}]

    def test_step_in_the_payload_wins_over_the_step_argument(self):
        logger = DataFrameLogger()
        logger.log_metrics({"loss": 0.5, "step": 9}, step=0)

        assert logger.train_logs_raw == [{"loss": 0.5, "step": 9}]

    def test_epoch_is_dropped(self):
        """Epochs carry no information: the training stream is unbounded."""
        logger = DataFrameLogger()
        logger.log_metrics({"epoch": 0, "loss": 0.5}, step=1)

        assert logger.train_logs_raw == [{"loss": 0.5, "step": 1}]

    def test_an_epoch_only_call_records_nothing(self):
        logger = DataFrameLogger()
        logger.log_metrics({"epoch": 0}, step=1)

        assert logger.train_logs_raw == []


class TestDataFrames:
    def test_train_df_is_indexed_by_step(self):
        logger = DataFrameLogger()
        logger.log_metrics({"loss": 0.5}, step=0)
        logger.log_metrics({"loss": 0.3}, step=1)

        df = logger.get_train_logs_df()

        assert isinstance(df, pd.DataFrame)
        assert df.index.name == "step"
        assert "step" not in df.columns
        assert list(df.index) == [0, 1]
        assert list(df["loss"]) == [0.5, 0.3]

    def test_empty_train_df(self):
        df = DataFrameLogger().get_train_logs_df()
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_val_df_for_one_id(self):
        logger = DataFrameLogger(val_ids=["valA"])
        logger.log_metrics({"valA_mmd": 0.5}, step=2)

        df = logger.get_val_logs_df("valA")

        assert len(df) == 1
        assert df.iloc[0]["valA_mmd"] == 0.5
        assert df.index.name == "step"

    def test_unknown_val_id_yields_an_empty_frame(self):
        df = DataFrameLogger(val_ids=["valA"]).get_val_logs_df("nope")
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_val_df_for_all_ids(self):
        logger = DataFrameLogger(val_ids=["valA", "valB"])
        logger.log_metrics({"valA_mmd": 0.5}, step=1)
        logger.log_metrics({"valB_mmd": 0.8}, step=1)

        result = logger.get_val_logs_df()

        assert set(result) == {"valA", "valB"}
        assert result["valA"].iloc[0]["valA_mmd"] == 0.5
        assert result["valB"].iloc[0]["valB_mmd"] == 0.8

    def test_val_ids_are_exposed_as_a_copy(self):
        logger = DataFrameLogger(val_ids=["valA"])
        logger.val_ids.append("mutated")
        assert logger.val_ids == ["valA"]
