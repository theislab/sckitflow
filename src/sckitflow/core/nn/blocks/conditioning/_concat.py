from typing import ClassVar

import torch
from scfit.registry import component

from sckitflow.core.nn.blocks.conditioning._base import BaseConditioningLayer, BaseConditioningLayerConfig

__all__ = [
    "ConcatConditioning",
    "ConcatConditioningConfig",
]


class ConcatConditioning(BaseConditioningLayer):
    """Class for concatenation based conditioning layers."""

    @property
    def output_dim(
        self,
    ) -> int:
        """Return the dimensionality of the conditioned output."""
        out_dim = self._latent_state_dim + self._latent_time_dim
        if self._latent_condition_dim is not None:
            out_dim = out_dim + self._latent_condition_dim
        return out_dim

    def forward(
        self,
        encoded_t: torch.Tensor,
        encoded_state: torch.Tensor,
        encoded_condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass on the conditioning layer, done by concatenating the inputs.

        :param encoded_t: The encoded representation for the current time index.
            Its trailing dimension should match what specified in the :attr: `self._latent_time_dim`
            attribute, otherwise a :class: `RuntimeError` is raised.
        :type encoded_t: class: `torch.Tensor`

        :param encoded_state: The encoded representation for the current states.
            Its trailing dimension should match what specified in the :attr: `self._latent_state_dim`
            attribute, otherwise a :class: `RuntimeError` is raised.
        :type encoded_state: class: `torch.Tensor`

        :param encoded_condition: Optional extra conditioning argument to be concatenated to the input.
            Its trailing dimension should match the corresponding one specified in the :attr: `self._latent_condition_dim`
            attribute, otherwise a :class: `RuntimeError` is raised.
        :type encoded_condition: class: `torch.Tensor`
        """
        return torch.cat([encoded_state, self._conditions(encoded_t, encoded_state, encoded_condition)], dim=-1)


@component("nn.conditioning.concat", builds=ConcatConditioning)
class ConcatConditioningConfig(BaseConditioningLayerConfig):
    layer: ClassVar[type[BaseConditioningLayer]] = ConcatConditioning
