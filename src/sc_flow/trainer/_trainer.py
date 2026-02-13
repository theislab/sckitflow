import logging
from collections.abc import Sequence
from typing import NewType

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from tqdm import tqdm

from sc_flow._runtime import (
    raise_runtime_error_on_backend_not_supported,
    set_jax_import_failed,
    set_torch_import_failed,
)

try:
    from torch import Generator, Tensor

    PRNG = NewType("PRNG", Generator)
except (ImportError, ModuleNotFoundError):
    set_torch_import_failed(True)
    Tensor = None

try:
    from jax import Array

    PRNG = NewType("PRNG", Array)
except (ImportError, ModuleNotFoundError):
    set_jax_import_failed(True)
    Array = None


MethodClass = NewType("MethodClass", None)
Callbacks = NewType("Callbacks", None)
DataLoader = NewType("DataLoader", None)

logger = logging.getLogger(__name__)


class FlowTrainer:
    """Abstract trainer for the supported methods.

    :param method: Method class
    :type method: class:

    :param callbacks: Callbacks class
    :type callbacks: class:

    :param _require_prng: Whether a Pseudo-Random Numbers Generator is required for the probability path.
        Pseudo-Random Numbers Generators are required for reproducibility of non-deterministic probability paths and should be instances of
        :class: `torch.Generator`, when provided. For non-deterministic probability paths a warning is displayed and it is set to the
        output of :constant: `torch.random.default_generator`.
    :type _require_prng: class: `bool`
    """

    def __init__(
        self,
        method: MethodClass,
        require_prng: bool,
        callbacks: Callbacks | None = None,
    ) -> None:
        self._method = method
        self._require_prng = require_prng
        self._callbacks = callbacks
        self._training_logs = {"loss": []}

    def _validation_step_on_single_condition(
        self,
        validation_batch: dict[str, Tensor | Array],
    ) -> dict:
        """Run validation on a single condition

        :param validation_batch: Validation batch containing one condition on which to perform a validation step
        :type validation_batch:

        Returns
        -------
            dict: The dictionary of matched predictions and targets to compute validation on
        """
        prediction = self._method.validation_step(validation_batch)
        target = validation_batch["target"]
        return {"target": target, "prediction": prediction}

    def _validation_step(
        self,
        validation_data: dict[str, dict],
    ) -> dict:
        """Run validation on all conditions

        :param validation_data: Validation dat with all the conditions on which to perform a validation
        :type validation_data:

        Returns
        -------
            dict: The dictionary of metrics computed on the all validation conditions
        """
        metrics = {}
        for condition, validation_batch in validation_data.items():
            validation_dict = self._validation_step_on_single_condition(validation_batch=validation_batch)
            metrics.update(self._callbacks.run_on_valid_step(validation_dict, condition=condition))
        return metrics

    def _train_step(
        self,
        batch: dict[str, Tensor | Array],
        prng_step_fn: PRNG | None = None,
    ) -> float:
        """Method that performs a training step

        :param batch:
        :type batch:

        :param prng_step_fn:
        :type prng_step_fn:

        Returns
        -------
            float: The value of the loss computed on a batch
        """
        loss = self._method.train_step(batch, prng_step_fn)
        return loss

    def _update_logs(
        self,
        metrics: dict[str, float],
    ) -> None:
        """Update training logs.

        :param metrics: Dictionary of metrics values to log
        :type metrics:
        """
        for metric_id, metric_val in metrics.items():
            if metric_id not in self._training_logs.keys():
                self._training_logs[metric_id] = []
            self._training_logs[metric_id].append(metric_val)

    def fit(
        self,
        train_dataloader: DataLoader,
        num_iterations: int,
        valid_freq: int,
        validation_dataloader: DataLoader | None = None,
        prng: Array | Generator | None = None,
    ) -> None:
        """Trains the model.

        :param train_dataloader: The train dataloader prepared by `model.prepare_train_data()`.
        :type train_batchtrain_dataloader_size: class:`DataLoader`

        :param num_iterations: The number of steps which to train the model on.
        :type num_iterations: class:`int`

        :param valid_freq: The number of gradient steps after which to perform a validation step,
            in case the validation data was prepared with `model.prerare_validation_data()`.
        :type valid_freq: class:`int | None`

        :param validation_dataloader: The validation dataloader prepared by `model.prepare_validation_data()`.
        :type validation_dataloader: class:`DataLoader`
        """
        from sc_flow._runtime import BACKEND

        if BACKEND == "torch":
            from torch import random
        elif BACKEND == "jax":
            from jax import random
        else:
            random = None
            raise_runtime_error_on_backend_not_supported(BACKEND)

        # sanity check when we require the prng
        if self._require_prng and (BACKEND == "torch") and (not isinstance(prng, Array)):
            msg = (
                f"The trainer {self.__class__.__name__} requires a PRNG. Please provide an instance of `Generator`"
                r"for reproducible results. Setting it to \`torch.random.default_generator\` by default."
            )
            logger.warning(msg)
            prng = random.default_generator
        elif self._require_prng and (BACKEND == "jax") and (not isinstance(prng, Array)):
            msg = (
                f"The trainer {self.__class__.__name__} requires a PRNG. Please provide an instance of `jax.Array`"
                r"for reproducible results. Setting it to \`torch.random.default_generator\` by default."
            )
            logger.warning(msg)
            prng = random.PRNGKey(0)
        elif (not self._require_prng) and (prng is not None):
            msg = f"PRNG provided to {self.__class__.__name__}, which is deterministic. Setting it to `None`."
            logger.warning(msg)
            prng = None

        pbar = tqdm(range(num_iterations))
        # prng_data = np.random.default_rng(0)

        do_validation = True
        if validation_dataloader is None:
            msg = "Validation step requires validation data. Please prepare validation data with `model.prepare_validation_data()` method."
            logger.warning(msg)
            do_validation = False

        for i in pbar:
            if BACKEND == "jax":
                print(type(prng))
                prng, prng_step_fn, prng_data = random.split(prng, 3)
            else:
                prng_step_fn = prng
                prng_data = prng
            batch = train_dataloader.sample(prng_data)
            loss = self._train_step(batch, prng_step_fn)

            self._update_logs({"loss": loss})

            if ((i + 1) % valid_freq == 0) and (i > 0) and do_validation:
                validation_batch = validation_dataloader.sample(prng_data)
                metrics = self._validation_step(validation_batch)
                self._update_logs(metrics)

    def plot_training_logs(
        self,
        figsize: Sequence[int] = (3, 3),
        keys_to_plot: str | Sequence[str] = "loss",
        show: bool = False,
    ) -> tuple[Figure, Axes]:
        """Method that plots the training logs.

        :param figsize: The output figure size.
        :type figsize: class:`Sequence[int]`

        :param keys_to_plot: The parameters specifying which keys in the `self.training_logs` are to be plotted. Default is `"loss"`
        :type num_iterations: class:`str | Sequence[str]`

        :param show: The parameter controlling whether to show the figure. Default is `False`
        :type show: class:`bool`
        """
        # handling keys to plot
        if isinstance(keys_to_plot, str):
            keys_to_plot = (keys_to_plot,)
        # sanity checks
        for key in keys_to_plot:
            msg = ""
            assert key in self._training_logs.keys(), msg
        # retrieving the logs we want to plot
        logs_to_plot = {log_id: log_data for log_id, log_data in self._training_logs.items() if log_id in keys_to_plot}

        fig, axes = plt.subplots(1, len(logs_to_plot), figsize=figsize)
        for idx, (loss_id, loss_history) in enumerate(logs_to_plot.items()):
            if len(logs_to_plot) == 1:
                current_axes = axes
            else:
                current_axes = axes[idx]
            current_axes.set_title(loss_id)
            current_axes.plot(loss_history)
        if show:
            fig.show()
        return fig, axes
