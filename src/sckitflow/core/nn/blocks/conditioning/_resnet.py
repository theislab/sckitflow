import torch
from scfit.registry import component

from sckitflow.core._utils import make_concatenation_possible
from sckitflow.core.nn.blocks._resnet import ResNet, ResNetConfig
from sckitflow.core.nn.blocks.conditioning._base import BaseConditioningLayer, BaseConditioningLayerConfig

__all__ = [
    "ResNetConditioning",
    "ResNetConditioningConfig",
]


class ResNetConditioning(BaseConditioningLayer):
    """Class for residual network based conditioning layers."""

    def __init__(
        self,
        *,
        latent_state_dim: int,
        latent_time_dim: int,
        latent_condition_dim: int | None = None,
        resnet_cfg: ResNetConfig | None = None,
    ) -> None:
        """Initializes the resnet based conditioning.

        :param latent_state_dim: The latent dimensionality of the input states.
        :type latent_state_dim: class: `int`

        :param latent_time_dim: The latent dimensionality of the input time index.
        :type latent_time_dim: class: `int`

        :param latent_condition_dim: (Optional) The dimensionality of extra conditioning argument to be concatenated to the input.
        :type latent_condition_dim: class: `int | None`

        :param resnet_cfg: Configuration object used to initialize the :class: `Resnet1d` object.
        :type resnet_cfg: class: `dict[str, Any]`
        """
        super().__init__(
            latent_state_dim,
            latent_time_dim,
            latent_condition_dim,
        )
        self._resnet_cfg = ResNetConfig() if resnet_cfg is None else resnet_cfg

        self._resnet: ResNet = self._resnet_cfg.build(
            self._latent_state_dim,
            self.embedding_dim,
        )

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
        to_concat = (make_concatenation_possible(encoded_t, encoded_state, -1),)
        if encoded_condition is not None:
            to_concat = to_concat + (make_concatenation_possible(encoded_condition, encoded_state, -1),)

        conditions = torch.concatenate(to_concat, dim=-1)
        return self._resnet(encoded_state, conditions)

    @property
    def output_dim(
        self,
    ) -> int:
        """Return the dimensionality of the conditioned output."""
        return self._resnet.output_dim

    @property
    def embedding_dim(
        self,
    ) -> int:
        """Return the dimensionality of the conditioned output."""
        embedding_dim = self._latent_time_dim
        if self._latent_condition_dim is not None:
            embedding_dim = embedding_dim + self._latent_condition_dim
        return embedding_dim


@component("nn.conditioning.resnet", builds=ResNetConditioning)
class ResNetConditioningConfig(BaseConditioningLayerConfig):
    resnet: ResNetConfig

    def build(
        self,
        *,
        latent_state_dim: int,
        latent_time_dim: int,
        latent_condition_dim: int | None = None,
    ) -> ResNet:
        return ResNetConditioning(
            latent_state_dim=latent_state_dim,
            latent_time_dim=latent_time_dim,
            latent_condition_dim=latent_condition_dim,
            resnet_cfg=self.resnet,
        )
