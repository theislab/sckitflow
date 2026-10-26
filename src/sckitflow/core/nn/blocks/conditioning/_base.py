import abc
from typing import ClassVar

import torch
from scfit.registry import Component

from sckitflow.core._utils import make_concatenation_possible

__all__ = ["BaseConditioningLayer", "BaseConditioningLayerConfig"]


class BaseConditioningLayer(abc.ABC, torch.nn.Module):
    """Base class for conditioning layers."""

    def __init__(
        self,
        latent_state_dim: int,
        latent_time_dim: int,
        latent_condition_dim: int | None = None,
    ) -> None:
        """Initializes the concatenation based conditioning.

        :param latent_state_dim: The latent dimensionality of the input states.
        :type latent_state_dim: class: `int`

        :param latent_time_dim: The latent dimensionality of the input time index.
        :type latent_time_dim: class: `int`

        :param latent_condition_dim: (Optional) The dimensionality of extra conditioning argument to be concatenated to the input.
        :type latent_condition_dim: class: `int | None`
        """
        super().__init__()
        self._latent_state_dim = latent_state_dim
        self._latent_time_dim = latent_time_dim
        self._latent_condition_dim = latent_condition_dim

    @property
    @abc.abstractmethod
    def output_dim(
        self,
    ) -> int:
        """Returns the dimensionality of the conditioned output."""

    @abc.abstractmethod
    def forward(
        self,
        encoded_t: torch.Tensor,
        encoded_state: torch.Tensor,
        encoded_condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Retrieves the encoded concatenation to be fed as input to the velocity field decoder.

        :param encoded_t: The time index at which the velocity field is computed.
        :type encoded_t: class: `torch.Tensor`

        :param encoded_state: The state at which the velocity fiel is computed.
        :type encoded_state: class: `torch.Tensor`

        :param encoded_condition: Optional extra conditioning argument to be concatenated to the input.
            Its trailing dimension should match the corresponding one specified in the :attr: `self._latent_condition_dim`
            attribute, otherwise a :class: `RuntimeError` is raised.
        :type encoded_condition: class: `torch.Tensor`
        """

    @property
    def embedding_dim(self) -> int:
        """Width of the conditioning signal: the time embedding plus the optional condition."""
        return self._latent_time_dim + (self._latent_condition_dim or 0)

    def _conditions(
        self,
        encoded_t: torch.Tensor,
        encoded_state: torch.Tensor,
        encoded_condition: torch.Tensor | None,
    ) -> torch.Tensor:
        """Checks the inputs, then broadcasts time and condition to the state's leading dims and concatenates them."""
        self._verify_inputs_shape(encoded_t, encoded_state, encoded_condition)
        parts = [encoded_t] if encoded_condition is None else [encoded_t, encoded_condition]
        return torch.cat([make_concatenation_possible(p, encoded_state, -1) for p in parts], dim=-1)

    def _verify_inputs_shape(
        self,
        encoded_t: torch.Tensor,
        encoded_state: torch.Tensor,
        encoded_condition: torch.Tensor | None = None,
    ) -> None:
        """Verifies that the inputs match their respective expected shapes."""
        if encoded_condition is None and self._latent_condition_dim is not None:
            raise RuntimeError(
                "When the latent condition dimension is specified, the corresponding input should be passed, found `None`."
            )
        for name, tensor, expected in (
            ("state", encoded_state, self._latent_state_dim),
            ("time", encoded_t, self._latent_time_dim),
            ("condition", encoded_condition, self._latent_condition_dim),
        ):
            if tensor is not None and tensor.shape[-1] != expected:
                raise RuntimeError(
                    f"The encoded {name} has the wrong trailing dimension. Found {tensor.shape[-1]}, expected {expected}"
                )

    @property
    def latent_state_dim(self) -> int:
        return self._latent_state_dim

    @property
    def latent_time_dim(self) -> int:
        return self._latent_time_dim

    @property
    def latent_condition_dim(self) -> int | None:
        return self._latent_condition_dim


class BaseConditioningLayerConfig(Component):
    layer: ClassVar[type[BaseConditioningLayer]]

    def build(
        self,
        *,
        latent_state_dim: int,
        latent_time_dim: int,
        latent_condition_dim: int | None = None,
    ) -> BaseConditioningLayer:
        return type(self).layer(
            latent_state_dim=latent_state_dim,
            latent_time_dim=latent_time_dim,
            latent_condition_dim=latent_condition_dim,
        )
