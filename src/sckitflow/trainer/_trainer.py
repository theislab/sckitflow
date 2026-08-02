from typing import Any

import lightning.pytorch as pl
import pandas as pd

from sckitflow.trainer._logger import DataFrameLogger

__all__ = ["Trainer"]


class Trainer(pl.Trainer):
    """Lightweight wrapper of :class:`lightning.pytorch.Trainer` with sckitflow defaults.

    Training is measured in *node-steps*: one node -- a single
    :class:`MatchedDistributions` drawn by the train sampler -- is one batch and one
    optimizer step. Because the sampler streams nodes without end there is only ever a
    single epoch, so both ``max_steps`` and ``val_check_interval`` count node-steps and
    epoch-based settings are switched off.

    :param n_train_steps: Number of node-steps to train for. Maps onto Lightning's
        ``max_steps``.
    :type n_train_steps: class: `int`

    :param valid_freq: How many node-steps between validation runs. Maps onto Lightning's
        ``val_check_interval``.
    :type valid_freq: class: `int`

    :param val_ids: Identifiers of the validation sets, in the same order as the
        ``val_dataloaders`` passed to :meth:`fit`. Used to name metrics and to split the
        recorded logs.
    :type val_ids: class: `list[str] | None`

    :param logger: Logger(s) to use. Defaults to `None`, in which case a
        :class:`DataFrameLogger` is installed so the logs are readable via
        :meth:`get_train_logs_df` without any further configuration.
    :type logger: class: `Any`

    :param kwargs: Other keyword arguments for :class:`lightning.pytorch.Trainer`.
    """

    def __init__(
        self,
        *,
        n_train_steps: int = 10_000,
        valid_freq: int = 1_000,
        val_ids: list[str] | None = None,
        logger: Any = None,
        enable_checkpointing: bool = False,
        num_sanity_val_steps: int = 0,
        log_every_n_steps: int = 1,
        **kwargs,
    ) -> None:
        self._df_logger = DataFrameLogger(val_ids=val_ids)
        if logger is None:
            logger = self._df_logger
        elif logger is False:
            # Explicitly opted out; the DataFrame accessors stay empty.
            pass
        elif isinstance(logger, list):
            logger = [*logger, self._df_logger]
        else:
            logger = [logger, self._df_logger]

        super().__init__(
            max_steps=n_train_steps,
            val_check_interval=valid_freq,
            # The training stream never ends, so epoch boundaries are meaningless:
            # validation has to be driven purely by `val_check_interval`.
            check_val_every_n_epoch=None,
            logger=logger,
            enable_checkpointing=enable_checkpointing,
            num_sanity_val_steps=num_sanity_val_steps,
            # Every node-step reaches the logger, matching the old hand-rolled loop which
            # recorded one row per node. Raise it to thin out long runs.
            log_every_n_steps=log_every_n_steps,
            **kwargs,
        )
        self._val_ids = list(val_ids) if val_ids else []

    @property
    def val_ids(self) -> list[str]:
        """Identifiers of the validation sets, positionally matching the val dataloaders.

        Read by :class:`MetricsCallback` to name the metrics it computes.
        """
        return list(self._val_ids)

    # -------------------------------------------------------------------------
    # Log retrieval and DataFrame conversion
    # -------------------------------------------------------------------------
    def get_train_logs_df(self) -> pd.DataFrame:
        """Return training logs as a pandas DataFrame."""
        return self._df_logger.get_train_logs_df()

    def get_val_logs_df(self, val_id: str | None = None) -> pd.DataFrame | dict[str, pd.DataFrame]:
        """Return validation logs as a pandas DataFrame.

        An unknown ``val_id`` yields an empty frame, mirroring the empty-logs case.
        """
        return self._df_logger.get_val_logs_df(val_id)

    @property
    def train_logs_raw(self) -> list[dict[str, Any]]:
        """Raw training logs (list of dicts)."""
        return self._df_logger.train_logs_raw

    @property
    def val_logs_raw(self) -> dict[str, list[dict[str, Any]]]:
        """Raw validation logs (dict of val_id -> list of dicts)."""
        return self._df_logger.val_logs_raw

    @property
    def current_step(self) -> int:
        """The number of node-steps taken so far."""
        return self.global_step
