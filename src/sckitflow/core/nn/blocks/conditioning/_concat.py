import torch
from scfit.registry import component

from sckitflow.core._utils import make_concatenation_possible
from sckitflow.core.nn.blocks.conditioning._base import BaseConditioningLayer, BaseConditioningLayerConfig

__all__ = [
    "ConcatConditioning",
    "ConcatConditioningConfig",
]


class ConcatConditioning(BaseConditioningLayer):
    """Class for concatenation based conditioning layers."""

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
        super().__init__(
            latent_state_dim,
            latent_time_dim,
            latent_condition_dim,
        )

        self._identity = torch.nn.Identity()

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
        # sanity checks
        self._verify_inputs_shape(
            encoded_t,
            encoded_state,
            encoded_condition,
        )

        # concatenating input
        to_concat = (encoded_state, make_concatenation_possible(encoded_t, encoded_state, -1))
        if encoded_condition is not None:
            to_concat = to_concat + (make_concatenation_possible(encoded_condition, encoded_state, -1),)
        concat_input = torch.concatenate(to_concat, dim=-1)
        return self._identity(concat_input)


@component("nn.conditioning.concat", builds=ConcatConditioning)
class ConcatConditioningConfig(BaseConditioningLayerConfig):
    def build(
        self,
        *,
        latent_state_dim: int,
        latent_time_dim: int,
        latent_condition_dim: int | None = None,
    ) -> ConcatConditioning:
        return ConcatConditioning(
            latent_state_dim=latent_state_dim,
            latent_time_dim=latent_time_dim,
            latent_condition_dim=latent_condition_dim,
        )
