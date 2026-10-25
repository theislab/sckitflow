import torch
from scfit.registry import component

from sckitflow.core._utils import make_concatenation_possible
from sckitflow.core.nn.blocks.conditioning._base import BaseConditioningLayer, BaseConditioningLayerConfig

__all__ = [
    "FiLMConditioning",
    "FiLMConditioningConfig",
]


class FiLMConditioning(BaseConditioningLayer):
    """Class for FiLM (Feature-wise Linear Modulation) based conditioning layers.

    The conditioning signal (time embedding + optional extra condition) is mapped
    to a pair of per-channel affine parameters ``(gamma, beta)`` via a linear
    projection followed by a SiLU non-linearity. The state is then modulated as
    ``gamma * encoded_state + beta``.
    """

    def __init__(
        self,
        *,
        latent_state_dim: int,
        latent_time_dim: int,
        latent_condition_dim: int | None = None,
    ) -> None:
        """Initializes the FiLM based conditioning.

        :param latent_state_dim: The latent dimensionality of the input states.
        :type latent_state_dim: class: `int`

        :param latent_time_dim: The latent dimensionality of the input time index.
        :type latent_time_dim: class: `int`

        :param latent_condition_dim: (Optional) The dimensionality of extra conditioning
            argument to be used in the FiLM modulation.
        :type latent_condition_dim: class: `int | None`
        """
        super().__init__(
            latent_state_dim,
            latent_time_dim,
            latent_condition_dim,
        )

        # Maps the conditioning signal to (gamma, beta), with SiLU non-linearity.
        self._linear = torch.nn.Linear(
            self.embedding_dim,
            2 * self._latent_state_dim,
        )
        self._activation = torch.nn.SiLU()

    @property
    def embedding_dim(self) -> int:
        """Return the dimensionality of the conditioning embedding."""
        embedding_dim = self._latent_time_dim
        if self._latent_condition_dim is not None:
            embedding_dim = embedding_dim + self._latent_condition_dim
        return embedding_dim

    @property
    def output_dim(self) -> int:
        """Return the dimensionality of the conditioned output."""
        return self._latent_state_dim

    def forward(
        self,
        encoded_t: torch.Tensor,
        encoded_state: torch.Tensor,
        encoded_condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass applying FiLM modulation to the encoded state.

        :param encoded_t: The encoded representation for the current time index.
        :type encoded_t: class: `torch.Tensor`

        :param encoded_state: The encoded representation for the current states.
        :type encoded_state: class: `torch.Tensor`

        :param encoded_condition: Optional extra conditioning argument used
            to compute the FiLM parameters.
        :type encoded_condition: class: `torch.Tensor`
        """
        # sanity checks
        self._verify_inputs_shape(
            encoded_t,
            encoded_state,
            encoded_condition,
        )

        # building the conditioning signal
        to_concat = (make_concatenation_possible(encoded_t, encoded_state, -1),)
        if encoded_condition is not None:
            to_concat = to_concat + (make_concatenation_possible(encoded_condition, encoded_state, -1),)
        conditions = torch.concatenate(to_concat, dim=-1)

        # (gamma, beta) generation with SiLU non-linearity
        gamma_beta = self._activation(self._linear(conditions))
        gamma, beta = gamma_beta.chunk(2, dim=-1)

        # FiLM modulation
        return gamma * encoded_state + beta


@component("nn.conditioning.film", builds=FiLMConditioning)
class FiLMConditioningConfig(BaseConditioningLayerConfig):
    def build(
        self,
        *,
        latent_state_dim: int,
        latent_time_dim: int,
        latent_condition_dim: int | None = None,
    ) -> FiLMConditioning:
        return FiLMConditioning(
            latent_state_dim=latent_state_dim,
            latent_time_dim=latent_time_dim,
            latent_condition_dim=latent_condition_dim,
        )
