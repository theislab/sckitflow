from tqdm import tqdm

from sc_flow.data.samplers._train import FTrainSampler
from sc_flow.data.samplers._validation import FValidationSampler
from sc_flow.methods._methods import BaseMethod
from sc_flow.methods._opt import BaseOptManager

__all__ = ["Trainer"]


class Trainer:
    """Trainer for the supported methods.

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
        method: BaseMethod,
        opt_manager: BaseOptManager,
    ) -> None:
        self._method = method
        self._opt_manager = opt_manager
        self._training_logs = {"loss": []}

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

    def _run_val_on_sampler(
        self,
        sampler: FValidationSampler,
        *args,
        **kwargs,
    ) -> None:
        predictions_list = []
        for node in sampler:
            target = node.target_distr
            preds = self._method.predict(node, *args, **kwargs)
            predictions_list.append[
                {
                    "target": target,
                    "preds": preds,
                }
            ]
        return predictions_list

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
        """Trains the model.

        :param train_sampler: The train dataloader prepared by `model.prepare_train_data()`.
        :type train_batchtrain_dataloader_size: class:`DataLoader`

        :param num_iterations: The number of steps which to train the model on.
        :type num_iterations: class:`int`

        :param valid_freq: The number of gradient steps after which to perform a validation step,
            in case the validation data was prepared with `model.prerare_validation_data()`.
        :type valid_freq: class:`int | None`

        :param validation_dataloader: The validation dataloader prepared by `model.prepare_validation_data()`.
        :type validation_dataloader: class:`DataLoader`
        """
        # flag for whether to run validation
        do_validation = True
        if val_samplers_dict is None:
            do_validation = False

        # iterate over training steps
        pbar = tqdm(range(n_train_steps))
        for train_step in pbar:
            # sample nodes and perform train step on them
            nodes = train_sampler.sample()
            for node in nodes:
                opt_data, step_dict = self._method.train_step(node)
                self._opt_manager.step(opt_data)
                self._update_logs(step_dict)

            # log information message
            if ((train_step + 1) % pbar_freq == 0) and (train_step > 0):
                msg = "| "
                for key, val in step_dict.items():
                    msg += f"{key}:{val} | "
                pbar.set_description(msg)

            # validation step
            if ((train_step + 1) % valid_freq == 0) and (train_step > 0) and do_validation:
                for val_id, val_sampler in val_samplers_dict.items():  # noqa
                    val_metrics = self._run_val_on_sampler(val_sampler)
                    self._update_logs(val_metrics)

    @property
    def opt_manager(self) -> BaseOptManager:
        return self._opt_manager
