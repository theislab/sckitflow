import torch
from scfit.registry import component

from sckitflow.core.nn.blocks.conditioning._base import BaseConditioningLayer, BaseConditioningLayerConfig

__all__ = [
    "AdaLNZeroConditioning",
    "AdaLNZeroConditioningConfig",
    "FiLMConditioning",
    "FiLMConditioningConfig",
]


class FiLMConditioning(BaseConditioningLayer):
    """Feature-wise linear modulation (FiLM, Perez et al. 2018) of the state by time and condition.

    A linear map of the conditioning signal gives a per-feature scale ``gamma`` and shift ``beta``, and the state
    becomes ``(1 + gamma) * encoded_state + beta``: ``gamma`` is an offset from 1, as in the paper's code. The
    projection starts at zero, as DiT's modulation does, so the layer starts as the identity and a random
    projection cannot rescale the state before training. The modulation is bilinear in condition and state,
    which concatenation cannot express. ``output_dim`` is ``latent_state_dim``.
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

        # one projection for both: its output splits into (gamma, beta); zero, so the layer starts as the identity
        self._linear = torch.nn.Linear(self.embedding_dim, 2 * self._latent_state_dim)
        torch.nn.init.zeros_(self._linear.weight)
        torch.nn.init.zeros_(self._linear.bias)

    @property
    def output_dim(self) -> int:
        """Return the dimensionality of the conditioned output."""
        return self._latent_state_dim

    def _modulate(self, conditions: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Scales and shifts ``state`` per feature by the ``(gamma, beta)`` projected from ``conditions``."""
        gamma, beta = self._linear(conditions).chunk(2, dim=-1)
        return (1 + gamma) * state + beta

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
        return self._modulate(self._conditions(encoded_t, encoded_state, encoded_condition), encoded_state)


class AdaLNZeroConditioning(FiLMConditioning):
    """adaLN-Zero (Peebles & Xie 2023): FiLM of the layer-normed state, with SiLU on the condition.

    The state is normalized without a learned scale or shift of its own and the conditioning signal goes through a
    SiLU before the projection; the projection starts at zero as in FiLM, so the layer starts as ``LayerNorm(state)``.
    DiT also gates a residual branch; this layer feeds the decoder directly, so there is no branch to gate.
    """

    def __init__(
        self,
        *,
        latent_state_dim: int,
        latent_time_dim: int,
        latent_condition_dim: int | None = None,
    ) -> None:
        """Initializes the adaLN-Zero based conditioning.

        :param latent_state_dim: The latent dimensionality of the input states.
        :type latent_state_dim: class: `int`

        :param latent_time_dim: The latent dimensionality of the input time index.
        :type latent_time_dim: class: `int`

        :param latent_condition_dim: (Optional) The dimensionality of extra conditioning
            argument to be used in the modulation.
        :type latent_condition_dim: class: `int | None`
        """
        super().__init__(
            latent_state_dim=latent_state_dim,
            latent_time_dim=latent_time_dim,
            latent_condition_dim=latent_condition_dim,
        )
        self._norm = torch.nn.LayerNorm(self._latent_state_dim, elementwise_affine=False)

    def forward(
        self,
        encoded_t: torch.Tensor,
        encoded_state: torch.Tensor,
        encoded_condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass modulating the layer-normed state.

        :param encoded_t: The encoded representation for the current time index.
        :type encoded_t: class: `torch.Tensor`

        :param encoded_state: The encoded representation for the current states.
        :type encoded_state: class: `torch.Tensor`

        :param encoded_condition: Optional extra conditioning argument used
            to compute the modulation.
        :type encoded_condition: class: `torch.Tensor`
        """
        conditions = torch.nn.functional.silu(self._conditions(encoded_t, encoded_state, encoded_condition))
        return self._modulate(conditions, self._norm(encoded_state))


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


@component("nn.conditioning.adaln_zero", builds=AdaLNZeroConditioning)
class AdaLNZeroConditioningConfig(BaseConditioningLayerConfig):
    def build(
        self,
        *,
        latent_state_dim: int,
        latent_time_dim: int,
        latent_condition_dim: int | None = None,
    ) -> AdaLNZeroConditioning:
        return AdaLNZeroConditioning(
            latent_state_dim=latent_state_dim,
            latent_time_dim=latent_time_dim,
            latent_condition_dim=latent_condition_dim,
        )
