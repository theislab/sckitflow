import torch

from sc_flow.backends.torch._types import MappedTensor
from sc_flow.backends.torch.nn._conditioning_layers import BaseConditioningLayer, get_conditioning_layer
from sc_flow.backends.torch.nn._vf import MLPVelocity

__all__ = ["MLPFlowMap"]


class MLPFlowMap(MLPVelocity):
    def _make_modules(self):
        nn = super()._make_modules()
        nn["s_encoder"] = self._make_time_encoder()
        return nn

    def _make_conditioning_layer(
        self,
    ) -> BaseConditioningLayer:
        """Initializes the conditioning layer according to the configurations specified during initialization."""
        tim_latent_dim_dim = self._time_encoder_output_dim if self._encode_time else self._get_num_time_features()
        return get_conditioning_layer(
            self._state_encoder_output_dim if self._encode_state else self._state_dim,
            2 * tim_latent_dim_dim,
            latent_condition_dim=self._conditioning_dim,
            conditioning_id=self._conditioning_id,
            conditioning_fn=self._conditioning_fn,
            conditioning_kwargs=self._conditioning_kwargs,
        )

    def forward(
        self,
        s: torch.Tensor,
        t: torch.Tensor,
        x: torch.Tensor,
        condition_dict: MappedTensor | None = None,
        source: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Performs a forward computation pass on the neural velocity field.

        :param t: The current time index at which the velocity field is computed.
        :type t: class: `torch.Tensor`

        :param x: The current state at which the velocity field is computed.
        :type x: class: `torch.Tensor`

        :param condition_dict: The input dictionary containing the data for
            each perturbation covariate.
        :type condition_dict: class: `MappedTensor`
        """
        encoded_xt = self._nn["state_encoder"](x)

        encoded_t = self._nn["time_features"](t)
        encoded_t = self._nn["t_encoder"](encoded_t)

        encoded_s = self._nn["time_features"](s)
        encoded_s = self._nn["t_encoder"](encoded_s)

        time_feats = torch.concatenate((encoded_s, encoded_t), dim=-1)

        encoded_condition = self._get_encoded_conditions(condition_dict)
        encoded_source = self._get_encoded_source(source)

        encoded_concat = self._nn["conditioning_layer"](
            time_feats, encoded_xt, encoded_condition=self._get_conditioning_input(encoded_condition, encoded_source)
        )
        return self._nn["vf_decoder"](encoded_concat)
