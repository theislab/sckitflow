import abc
from typing import Any

import torch

from sc_flow.backends.torch._data_utils import (
    extract_step_data,
    prepare_latent_inference,
    write_continuous_cond_cov_to_step_data,
)
from sc_flow.backends.torch._types import StepData
from sc_flow.backends.torch.methods._base import TorchBaseMethod, TorchGenerativeFlow
from sc_flow.data._composite import MatchedData

__all__ = ["SurrogateModel", "SurrogateFlowModel"]


class SurrogateModel(abc.ABC, torch.nn.Module):
    """Base class for differentiable wrappers around modules.

    This family of objects will be used to call models as surrogates for inverse problems.
    """

    def __init__(
        self,
        condition_key: str,
        model: TorchBaseMethod,
        *args,
        base_data: MatchedData | None = None,
        model_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initializes the wrapper from the given model.

        :param condition_key: The condition key to override when querying the surrogate model.
        :type condition_key: class: `str`

        :param model: The surrogate model to wrap around.
        :type model: class: `torch.nn.Module`

        :param base_data: Optional base data used as context. Defaults to `None`.
        :type base_data: class: `MatchedData | None`
        """
        super().__init__()
        self._condition_key = condition_key
        self._model = model
        self._base_data = base_data
        self._model_kwargs = {} if model_kwargs is None else model_kwargs

    def _get_step_data(self, x: torch.Tensor) -> StepData:
        """"""  # noqa
        return write_continuous_cond_cov_to_step_data(
            self._condition_key, x, base_data=self._base_data, dtype=x.dtype, device=x.device
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluates the model from the input continuous condition covariates.

        :param x: Tensor containing the continuous conditions covariates which to evaluate the surrogate model on.
        :type x: class: `torch.Tensor`
        """
        # ---- 1. Get updated step data ----
        step_data = self._get_step_data(x)

        # ---- 2. Predict with surrogate model ----
        return self.model.predict(
            step_data,
            no_grad=False,
            **self._model_kwargs,
        )

    @property
    def condition_key(self) -> str:
        """The condition key to optimize over."""
        return self._condition_key

    @property
    def model(self) -> torch.nn.Module:
        """The underlying surrogate model."""
        return self._model

    @property
    def base_data(self) -> MatchedData | None:
        """"""  # noqa
        return self._base_data


class SurrogateFlowModel(SurrogateModel):
    def __init__(
        self,
        condition_key: str,
        model: TorchGenerativeFlow,
        *args,
        base_data: MatchedData | None = None,
        model_kwargs: dict[str, Any] | None = None,
        fix_latent: bool = True,
        n_fwd_samples: int | None = None,
    ) -> None:
        super().__init__(condition_key, model, *args, base_data=base_data, model_kwargs=model_kwargs)

        self._fix_latent = fix_latent
        self._n_fwd_samples = n_fwd_samples

        self._latent: torch.Tensor | None = self._presample_latent()

    def _presample_latent(self) -> torch.Tensor | None:
        """"""  # noqa
        if not self._fix_latent:
            return None

        step_data = extract_step_data(self._base_data)
        return prepare_latent_inference(
            step_data.source_state,
            step_data.target_state,
            self._model.noise_sampler,
            n_samples=self._n_fwd_samples,
            generate_from_noise=self._model._generate_from_noise,
            dtype=self._model._dtype,
            device=self._model._device_id,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """"""  # noqa
        # ---- 1. Get step data ----
        step_data = self._get_step_data(x)

        # ---- 2. Override dictionary to remove latent ----
        model_kwargs = self._model_kwargs.copy()
        if self._fix_latent:
            if "latent" in model_kwargs:
                model_kwargs.pop("latent")
            latent = self._latent
        else:
            latent = model_kwargs.pop("latent", None)

        # ---- 3. Predict with surrogate model ----
        return self.model.predict(
            step_data,
            no_grad=False,
            latent=latent,
            **model_kwargs,
        )
