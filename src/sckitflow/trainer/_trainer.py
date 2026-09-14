from collections.abc import Iterable, Sequence
from typing import Any

import pandas as pd
from tqdm import tqdm

from sckitflow.core._types import StepData
from sckitflow.core.methods._base import SupportsInference, SupportsTraining
from sckitflow.core.methods._opt import OptimizationManager
from sckitflow.trainer._callbacks import BaseCallback, TrainingCallbacks

__all__ = ["Trainer"]


class Trainer:
    """Trainer driving a training protocol and, optionally, an inference protocol.

    :param training_protocol: Anything satisfying `SupportsTraining` (a
        `BaseTrainingProtocol`, a `BaseFlowTrainingProtocol`, a
        `MatchedTrainingProtocol` wrapper, or a user-defined class with the
        same surface). `compute_loss` is called once per streamed batch.
    :param opt_manager: Optimization manager stepping the module.
    :param inference_protocol: Anything satisfying `SupportsInference`. Used
        during validation only; when `None`, validation is skipped entirely.
    :param callbacks: Either a `TrainingCallbacks` instance, a sequence of
        `BaseCallback`, or `None`. If a sequence is given, it is wrapped in a
        `TrainingCallbacks`.
    """

    def __init__(
        self,
        training_protocol: SupportsTraining,
        opt_manager: OptimizationManager,
        inference_protocol: SupportsInference | None = None,
        callbacks: TrainingCallbacks | Sequence[BaseCallback] | None = None,
    ) -> None:
        self._training_protocol = training_protocol
        self._inference_protocol = inference_protocol
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

    def _run_val_on_loader(
        self,
        loader: Iterable[StepData],
        val_id: str,
        *,
        predict_kwargs: dict[str, Any] | None = None,
        cb_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Run validation over one loader (one finite pass) and store predictions.

        ``loader`` is any iterable yielding ready ``StepData`` batches. The two
        kwarg dicts go to different places and are kept apart: ``predict_kwargs``
        to the inference protocol's ``predict`` (e.g. CFM's ``n_samples``),
        ``cb_kwargs`` to the callbacks.

        When no inference protocol is configured, this is a no-op.
        """
        # early return when no inference protocol is provided
        if self._inference_protocol is None:
            return None

        predict_kwargs = {} if predict_kwargs is None else predict_kwargs
        cb_kwargs = {} if cb_kwargs is None else cb_kwargs
        predictions_dict = {}
        for node_id, step_data in enumerate(loader):
            # The loader yields ready `StepData`; the ground-truth target is its
            # `target_state` tensor (the metric callbacks accept tensors directly).
            target_array = step_data["target_state"]
            preds = self._inference_protocol.predict(step_data, **predict_kwargs)
            # Extract the actual data from the prediction object
            if hasattr(preds, "X"):
                preds_array = preds.X
            else:
                preds_array = preds
            predictions_dict[str(node_id)] = {"predictions": preds_array, "targets": target_array}

        # Trigger callbacks with the predictions dictionary
        metrics_dict = self._callbacks.on_valid_step(self, self._current_step, val_id, predictions_dict, **cb_kwargs)
        metrics_dict.update({"step": self._current_step})

        # Store the metrics computed by the callbacks, tagged with the validation step.
        self._append_val_log(val_id, metrics_dict)

    def _get_logs_df(self, logs: list[dict[str, Any]] | None) -> pd.DataFrame:
        """Return logs as a pandas DataFrame indexed by training step.

        `step` becomes the index rather than a column, so the remaining columns
        are all metrics and plot against the step count directly.
        """
        if logs is None:
            return pd.DataFrame()
        log_df = pd.DataFrame(logs)
        if "step" in log_df.columns:
            idx = log_df["step"]
            log_df.index = idx
            log_df.index.name = "step"
            # `columns=` already implies the axis; passing both is a pandas error.
            log_df = log_df.drop(columns=["step"])
        return log_df

    def train(
        self,
        train_loader: Iterable[StepData],
        *,
        val_loaders: dict[str, Iterable[StepData]] | None = None,
        valid_freq: int = 1_000,
        pbar_freq: int = 100,
        compute_loss_kwargs: dict[str, Any] | None = None,
        val_predict_kwargs: dict[str, Any] | None = None,
        cb_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Trains by iterating ``train_loader`` -- its length is the number of steps.

        The caller sizes ``train_loader`` (e.g. ``Loader.set_n_iters(n_train_steps)``);
        the trainer just does one gradient step per streamed ``StepData`` and
        validates every ``valid_freq`` steps.

        :param train_loader: Anything yielding ready ``StepData``; its length is
            the step count.
        :type train_loader: class: `Iterable[StepData]`

        :param val_loaders: One ``{val_id: loader}`` per validation set, or
            ``None`` to skip validation. Defaults to `None`.
        :type val_loaders: class: `dict[str, Iterable[StepData]] | None`

        :param valid_freq: Run validation every this many training steps.
            Defaults to ``1_000``.
        :type valid_freq: class: `int`

        :param pbar_freq: Refresh the progress-bar description every this many
            steps. Defaults to ``100``.
        :type pbar_freq: class: `int`

        :param compute_loss_kwargs: Forwarded to the training protocol's
            ``compute_loss``. Method-specific. Defaults to `None`.
        :type compute_loss_kwargs: class: `dict[str, Any] | None`

        :param val_predict_kwargs: Forwarded to the inference protocol's
            ``predict`` during validation -- e.g. CFM's ``n_samples``, required
            when the model generates from noise. Defaults to `None`.
        :type val_predict_kwargs: class: `dict[str, Any] | None`

        :param cb_kwargs: Forwarded to every callback hook. Defaults to `None`.
        :type cb_kwargs: class: `dict[str, Any] | None`
        """
        do_validation = val_loaders is not None

        # Each dict goes to exactly one destination -- training step, inference, callbacks.
        compute_loss_kwargs = {} if compute_loss_kwargs is None else compute_loss_kwargs
        val_predict_kwargs = {} if val_predict_kwargs is None else val_predict_kwargs
        cb_kwargs = {} if cb_kwargs is None else cb_kwargs

        # Call on_train_begin
        self._callbacks.on_train_begin(self, **cb_kwargs)

        # One gradient step per streamed `StepData`; the loader defines how many.
        pbar = tqdm(train_loader)
        for step_data in pbar:
            self._current_step += 1
            opt_data, step_dict = self._training_protocol.compute_loss(step_data, **compute_loss_kwargs)
            step_dict.update({"step": self._current_step})
            self._opt_manager.step(opt_data)
            self._append_train_log(step_dict)

            # Call on_train_step with the step's metrics
            self._callbacks.on_train_step(self, self._current_step, step_dict, **cb_kwargs)

            # Update progress bar description
            if self._current_step % pbar_freq == 0:
                msg = "| " + " | ".join(
                    f"{k}:{v:.4f}" if isinstance(v, float) else f"{k}:{v}" for k, v in step_dict.items()
                )
                pbar.set_description(msg)

            # Validation step
            if self._current_step % valid_freq == 0 and do_validation:
                for val_id, val_loader in val_loaders.items():
                    self._run_val_on_loader(val_loader, val_id, predict_kwargs=val_predict_kwargs, cb_kwargs=cb_kwargs)

        # Call on_train_end
        self._callbacks.on_train_end(self, **cb_kwargs)

    # -------------------------------------------------------------------------
    # Log retrieval and DataFrame conversion
    # -------------------------------------------------------------------------
    def get_train_logs_df(self) -> pd.DataFrame:
        """Return training logs as a pandas DataFrame."""
        return self._get_logs_df(self._train_logs)

    def get_val_logs_df(self, val_id: str | None = None) -> pd.DataFrame | dict[str, pd.DataFrame]:
        """Return validation logs as a pandas DataFrame.

        An unknown ``val_id`` yields an empty frame, mirroring the empty-logs case.
        """
        if val_id is not None:
            # `.get` so an unknown id yields an empty frame instead of a KeyError.
            return self._get_logs_df(self._val_logs.get(val_id))
        return {vid: self._get_logs_df(logs) for vid, logs in self._val_logs.items()}

    @property
    def training_protocol(self) -> SupportsTraining:
        """The training protocol: anything satisfying `SupportsTraining`."""
        return self._training_protocol

    @property
    def inference_protocol(self) -> SupportsInference | None:
        """The inference protocol, or `None` when validation is disabled."""
        return self._inference_protocol

    @property
    def train_logs_raw(self) -> list[dict[str, Any]]:
        """Raw training logs (list of dicts)."""
        return self._train_logs

    @property
    def val_logs_raw(self) -> dict[str, list[dict[str, Any]]]:
        """Raw validation logs (dict of val_id -> list of dicts)."""
        return self._val_logs

    @property
    def opt_manager(self) -> OptimizationManager:
        return self._opt_manager

    @property
    def current_step(self) -> int:
        return self._current_step
