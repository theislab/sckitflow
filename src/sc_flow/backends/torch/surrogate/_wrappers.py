import abc

import torch

from sc_flow.backends.torch.methods._utils import StepData  # move it to types

__all__ = ["BaseSurrogateWrapper", "GenerativeFlowSurrogateWrapper"]


class BaseSurrogateWrapper(abc.ABC, torch.nn.Module):
    """Base class for differentiable wrappers around modules.

    This family of objects will be used to call models as surrogates for inverse problems.
    """

    def __init__(self, model: torch.nn.Module, *args) -> None:
        """Initializes the wrapper from the given model.

        :param model: The surrogate model to wrap around.
        :type model: class: `torch.nn.Module`
        """
        super().__init__()
        self._model = model

    def _write_input_to_step_data(self, x: torch.Tensor) -> StepData:
        """"""  # noqa

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluates the model from the input continuous condition covariates.

        :param x: Tensor containing the continuous conditions covariates which to evaluate the surrogate model on.
        :type x: class: `torch.Tensor`
        """


class GenerativeFlowSurrogateWrapper(BaseSurrogateWrapper):
    """Wrapper class for wrapper around flow-based generative models."""

    def __init__(self, model: torch.nn.Module, *args) -> None:
        """Initializes the wrapper from the given model.

        :param model: The surrogate model to wrap around.
        :type model: class: `torch.nn.Module`
        """
        super().__init__(model)
