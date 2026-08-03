from typing import Any

import pandas as pd
import torch
from lightning.pytorch.loggers.logger import Logger
from lightning.pytorch.utilities import rank_zero_only

__all__ = ["DataFrameLogger"]


class DataFrameLogger(Logger):
    """In-memory logger that keeps every logged metric as a row, indexed by step.

    The zero-configuration way to read a run back: no files are written and no external
    service is contacted, so :meth:`Trainer.get_train_logs_df` can hand back a
    :class:`pandas.DataFrame` straight after :meth:`fit`. Pass a real
    :mod:`lightning.pytorch.loggers` logger instead (or alongside) to also ship metrics
    to disk or to W&B.

    Rows are split into training and validation by metric name: a metric is validation
    data when its name starts with ``"{val_id}_"`` for one of the ``val_ids`` given at
    construction. That is the naming :class:`MetricsCallback` emits.
    """

    def __init__(self, val_ids: list[str] | None = None) -> None:
        """Initializes the logger.

        :param val_ids: Identifiers of the validation sets in play, used to route metrics
            to the right frame. Defaults to `None`, meaning every metric is training data.
        :type val_ids: class: `list[str] | None`
        """
        super().__init__()
        self._val_ids = list(val_ids) if val_ids else []
        self._train_rows: list[dict[str, Any]] = []
        self._val_rows: dict[str, list[dict[str, Any]]] = {val_id: [] for val_id in self._val_ids}

    @property
    def name(self) -> str:
        return "DataFrameLogger"

    @property
    def version(self) -> int:
        return 0

    @property
    def val_ids(self) -> list[str]:
        """Exposes the :param val_ids: attribute set at initialization."""
        return list(self._val_ids)

    @staticmethod
    def _unwrap(value: Any) -> Any:
        return value.item() if isinstance(value, torch.Tensor) else value

    def _owning_val_id(self, metric: str) -> str | None:
        """Returns the validation id a metric name belongs to, or `None` when it is training data."""
        for val_id in self._val_ids:
            if metric.startswith(f"{val_id}_"):
                return val_id
        return None

    @rank_zero_only
    def log_hyperparams(self, params: dict[str, Any], *args, **kwargs) -> None:
        """Hyperparameters are not recorded; only metrics end up in the frames."""

    @rank_zero_only
    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Records one row per `log_metrics` call, partitioned by validation id."""
        # Lightning appends `epoch` to every call; it carries no information here because
        # the unbounded training stream means there is only ever one epoch.
        metrics = {k: v for k, v in metrics.items() if k != "epoch"}
        if not metrics:
            return

        step = metrics.pop("step", step)

        # A single call can mix training and validation metrics, so bucket by name and
        # emit at most one row per destination.
        buckets: dict[str | None, dict[str, Any]] = {}
        for metric, value in metrics.items():
            val_id = self._owning_val_id(metric)
            buckets.setdefault(val_id, {})[metric] = self._unwrap(value)

        for val_id, row in buckets.items():
            row["step"] = self._unwrap(step)
            if val_id is None:
                self._train_rows.append(row)
            else:
                self._val_rows[val_id].append(row)

    @staticmethod
    def _to_df(rows: list[dict[str, Any]] | None) -> pd.DataFrame:
        """Returns rows as a DataFrame indexed by training step.

        ``step`` becomes the index rather than a column, so the remaining columns are all
        metrics and plot against the step count directly.
        """
        if not rows:
            return pd.DataFrame()
        log_df = pd.DataFrame(rows)
        if "step" in log_df.columns:
            idx = log_df["step"]
            log_df.index = idx
            log_df.index.name = "step"
            # `columns=` already implies the axis; passing both is a pandas error.
            log_df = log_df.drop(columns=["step"])
        return log_df

    def get_train_logs_df(self) -> pd.DataFrame:
        """Returns the training metrics as a DataFrame indexed by step."""
        return self._to_df(self._train_rows)

    def get_val_logs_df(self, val_id: str | None = None) -> pd.DataFrame | dict[str, pd.DataFrame]:
        """Returns validation metrics as a DataFrame, or one per validation id.

        An unknown ``val_id`` yields an empty frame, mirroring the empty-logs case.
        """
        if val_id is not None:
            return self._to_df(self._val_rows.get(val_id))
        return {vid: self._to_df(rows) for vid, rows in self._val_rows.items()}

    @property
    def train_logs_raw(self) -> list[dict[str, Any]]:
        """Raw training logs (list of dicts)."""
        return self._train_rows

    @property
    def val_logs_raw(self) -> dict[str, list[dict[str, Any]]]:
        """Raw validation logs (dict of val_id -> list of dicts)."""
        return self._val_rows
