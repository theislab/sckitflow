from collections.abc import Sequence
from typing import Any

import pandas as pd
from tqdm import tqdm

from sc_flow.data.samplers._train import FTrainSampler
from sc_flow.data.samplers._validation import FValidationSampler
from sc_flow.methods._methods import BaseMethod
from sc_flow.methods._opt import BaseOptManager
from sc_flow.trainer._callbacks import BaseCallback, TrainingCallbacks

__all__ = ["Trainer"]


class Trainer:
    """Trainer for the supported methods.

    :param method: Method class
    :param opt_manager: Optimization manager
    :param callbacks: Either a TrainingCallbacks instance, a list of BaseCallback,
        or None. If a list is given, it will be wrapped in a TrainingCallbacks.
    """

    def __init__(
        self,
        method: BaseMethod,
        opt_manager: BaseOptManager,
        callbacks: TrainingCallbacks | Sequence[BaseCallback] | None = None,
    ) -> None:
        self._method = method
        self._opt_manager = opt_manager

        # Normalize callbacks to a TrainingCallbacks instance
        if callbacks is None:
            self._callbacks = TrainingCallbacks([])
        elif isinstance(callbacks, TrainingCallbacks):
            self._callbacks = callbacks
        elif isinstance(callbacks, Sequence):
            self._callbacks = TrainingCallbacks(callbacks)
        else:
            raise TypeError(
                f"callbacks must be a TrainingCallbacks, a sequence of BaseCallback, or None, got {type(callbacks)}"
            )

        # Training logs: list of dicts, each dict contains metrics for one training step
        self._train_logs: list[dict[str, Any]] = []
        # Validation logs: dict mapping val_id to list of dicts (one per validation run)
        self._val_logs: dict[str, list[dict[str, Any]]] = {}

        # set current step
        self._current_step = 0

    def _append_train_log(self, log_dict: dict[str, Any]) -> None:
        self._train_logs.append(log_dict)

    def _append_val_log(self, val_id: str, log_dict: dict[str, Any]) -> None:
        if val_id not in self._val_logs:
            self._val_logs[val_id] = []
        self._val_logs[val_id].append(log_dict)

    def _run_val_on_sampler(
        self,
        sampler: FValidationSampler,
        val_id: str,
        step: int,
        *args,
        **kwargs,
    ) -> None:
        """Run validation on a sampler and store predictions."""
        predictions_dict = {}
        for node in sampler:
            target = self._method.extract_state_data(node.target_distr.state_data)
            preds = self._method.predict(node, *args, **kwargs)
            # Extract the actual data from PredictionData object
            if hasattr(preds, "samples"):
                preds_array = preds.samples
            else:
                preds_array = preds
            if hasattr(target, "X"):
                target_array = target.X
            else:
                target_array = target
            node_id = str(node)  # or use a better identifier
            predictions_dict[node_id] = {"predictions": preds_array, "targets": target_array}

        # Trigger callbacks with the predictions dictionary
        metrics_dict = self._callbacks.on_valid_step(self, step, val_id, predictions_dict, **kwargs)
        metrics_dict.update({"step": self._current_step})

        # Store raw predictions in logs
        self._append_val_log(val_id, metrics_dict)

    def _get_logs_df(self, logs_dict: dict[str, Any] | None) -> pd.DataFrame:
        """Return training logs as a pandas DataFrame."""
        if logs_dict is None:
            return pd.DataFrame()
        log_df = pd.DataFrame(logs_dict)
        if "step" in log_df.columns:
            idx = log_df["step"]
            log_df.index = idx
            log_df.index.name = "step"
            log_df = log_df.drop(columns=["step"], axis=1)
        return log_df

    def train(
        self,
        train_sampler: FTrainSampler,
        val_samplers_dict: dict[str, FValidationSampler] | None = None,
        n_train_steps: int = 10_000,
        valid_freq: int = 1_000,
        pbar_freq: int = 100,
        *args,
        **kwargs,
    ) -> None:
        """Trains the model."""
        do_validation = val_samplers_dict is not None

        # Call on_train_begin
        self._callbacks.on_train_begin(self, **kwargs)

        pbar = tqdm(range(n_train_steps))
        for self._current_step in pbar:
            # Sample nodes and perform training steps
            nodes = train_sampler.sample()
            step_dict = {}
            for node in nodes:
                opt_data, step_dict = self._method.train_step(node)
                step_dict.update({"step": self._current_step})
                self._opt_manager.step(opt_data)
                self._append_train_log(step_dict)

            # Call on_train_step with the step_dict from the last node
            self._callbacks.on_train_step(self, self._current_step, step_dict, **kwargs)

            # Update progress bar description
            if ((self._current_step + 1) % pbar_freq == 0) and (self._current_step > 0):
                msg = "| " + " | ".join(
                    f"{k}:{v:.4f}" if isinstance(v, float) else f"{k}:{v}" for k, v in step_dict.items()
                )
                pbar.set_description(msg)

            # Validation step
            if ((self._current_step + 1) % valid_freq == 0) and (self._current_step > 0) and do_validation:
                for val_id, val_sampler in val_samplers_dict.items():
                    self._run_val_on_sampler(val_sampler, val_id, self._current_step, *args, **kwargs)

        # Call on_train_end
        self._callbacks.on_train_end(self, **kwargs)

    # -------------------------------------------------------------------------
    # Log retrieval and DataFrame conversion
    # -------------------------------------------------------------------------
    def get_train_logs_df(self) -> pd.DataFrame:
        """Return training logs as a pandas DataFrame."""
        return self._get_logs_df(self._train_logs)

    def get_val_logs_df(self, val_id: str | None = None) -> pd.DataFrame | dict[str, pd.DataFrame]:
        """Return validation logs as a pandas DataFrame."""
        if val_id is not None:
            return self._get_logs_df(self._val_logs[val_id])
        return {vid: self._get_logs_df(logs) for vid, logs in self._val_logs.items()}

    @property
    def train_logs_raw(self) -> list[dict[str, Any]]:
        """Raw training logs (list of dicts)."""
        return self._train_logs

    @property
    def val_logs_raw(self) -> dict[str, list[dict[str, Any]]]:
        """Raw validation logs (dict of val_id -> list of dicts)."""
        return self._val_logs

    @property
    def opt_manager(self) -> BaseOptManager:
        return self._opt_manager

    @property
    def current_step(self) -> int:
        return self._current_step
