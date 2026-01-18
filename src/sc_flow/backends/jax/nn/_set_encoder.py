from typing import Any, Literal

import jax.numpy as jnp
from flax import linen as nn

from sc_flow._types import LayersDict, NestedLayersDict
from sc_flow.backends.jax._types import MappedArray
from sc_flow.backends.jax.nn._modules import BaseModule, FunctionalModule
from sc_flow.backends.jax.nn._utils import init_module_from_dict

__all__ = ["SetEncoder"]


class SetEncoder(BaseModule):
    """Encoder for set of conditioning covariates."""

    input_layers: NestedLayersDict
    output_dim: int
    pooling_mode: Literal["mean", "sum"] = "mean"
    pooling_kwargs: dict[str, Any] | None = None
    output_layers_kwargs: LayersDict | None = None

    def setup(self) -> None:
        """Initializes the set encoder modules."""
        _pooling_kwargs = {} if self.pooling_kwargs is None else self.pooling_kwargs
        output_layers_kwargs = {} if self.output_layers_kwargs is None else self.output_layers_kwargs

        self.covariate_ids = list(self.input_layers.keys())
        self.input_encoder_list = [
            init_module_from_dict(covariate_layers_dict) for covariate_layers_dict in self.input_layers.values()
        ]

        self.pooling_layer = self._make_pooling_layer()
        self.output_layer = init_module_from_dict(
            output_layers_kwargs, input_dim=self.decoder_input_dim, output_dim=self.output_dim
        )

    def _make_pooling_layer(self) -> nn.Module:
        """Initializes the pooling layer."""
        if self.pooling_mode == "mean":
            pooling_fn = lambda x: jnp.mean(x, axis=-2)
            return FunctionalModule(pooling_fn)
        elif self.pooling_mode == "sum":
            pooling_fn = lambda x: jnp.sum(x, axis=-2)
            return FunctionalModule(pooling_fn)
        else:
            msg = f'Pooling mode {self.pooling_mode} is not supported, possible options are `["mean", "sum"]`'
            raise ValueError(msg)

    def __call__(
        self,
        condition_dict: MappedArray,
    ) -> jnp.ndarray:
        """Forward computation pass on the set encoder.

        :param condition_dict: The input dictionary containing the data for
            each perturbation covariate.
        :type condition_dict: class: `MappedArray`
        """
        encoded_covariates = {}
        for covariate_id in condition_dict.keys():
            if covariate_id not in self.covariate_ids:
                msg = f"Input encoder not found for covariate {covariate_id}"
                raise KeyError(msg)
            encoded_covariates[covariate_id] = self.input_encoder_list[self.covariate_ids.index(covariate_id)](
                condition_dict[covariate_id]
            )

        pooled_covariates = jnp.concatenate(tuple(encoded_covariates.values()), axis=-1)
        pooled_covariates = self.pooling_layer(pooled_covariates)
        return self.output_layer(pooled_covariates)

    @property
    def decoder_input_dim(
        self,
    ) -> int:
        """Retrieves the input dimensionality for the output decoder."""
        # summing up all the dimensions of non pooled covariates
        decoder_input_dim = sum(layer_dict["output_dim"] for layer_dict in self.input_layers.values())
        return decoder_input_dim
