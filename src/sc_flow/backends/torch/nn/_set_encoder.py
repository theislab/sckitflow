from typing import Any, Literal

import torch

from sc_flow._types import LayersDict, NestedLayersDict
from sc_flow.backends.torch._types import MappedTensor
from sc_flow.backends.torch.nn._modules import BaseModule, FunctionalModule
from sc_flow.backends.torch.nn._utils import init_module_from_dict

__all__ = ["SetEncoder"]


class SetEncoder(BaseModule):
    """Encoder for set of conditioning covariates."""

    def __init__(
        self,
        input_layers: NestedLayersDict,
        output_dim: int,
        pooling_mode: Literal["mean", "sum"] = "mean",
        pooling_kwargs: dict[str, Any] | None = None,
        output_layers_kwargs: LayersDict | None = None,
    ) -> None:
        """Initializes the set encoder.

        :param input_layers: Dictionary mapping each perturbation covariate
            identifier to the configurations for their respective input layer.
        :type input_layers: class: `NestedLayersDict`

        :param output_dim: The output dimensionality of the set encoder.
        :type output_dim: class: `int`

        :param pooling_mode: Identifier for the pooling strategy of conditioning covariates.
            Defaults to `"mean"`.
        :type pooling_mode: class: `Literal["mean", "sum"]`

        :param pooling_kwargs: Optional keyword arguments for pooling layer.
            Ignored when pooling is `"mean"` or `"sum"`, defaults to `None`.
        :type pooling_kwargs: class: `dict[str, Any]`

        :param output_layers_kwargs: Dictionary containing the configurations for the output layer.
            Defaults to `None`.
        :type output_layers_kwargs: class: `LayersDict | None`
        """
        super().__init__()
        self._input_layers = input_layers
        self._output_dim = output_dim
        self._pooling_mode = pooling_mode
        self._pooling_kwargs = {} if pooling_kwargs is None else pooling_kwargs
        self._output_layers_kwargs = {} if output_layers_kwargs is None else output_layers_kwargs

        self._condition_encoder = self._make_modules()

    def _make_input_layers(
        self,
    ) -> dict[str, torch.nn.Module]:
        """Initializes the input layers."""
        layers = {}
        for covariate_id, covariate_layers_dict in self._input_layers.items():
            layers[covariate_id] = init_module_from_dict(covariate_layers_dict)
        return layers

    def _make_pooling_layer(
        self,
    ) -> torch.nn.Module:
        """Initializes the pooling layer."""
        if self._pooling_mode == "mean":
            pooling_fn = lambda x: torch.mean(x, dim=-2)
            return FunctionalModule(pooling_fn)
        elif self._pooling_mode == "sum":
            pooling_fn = lambda x: torch.sum(x, dim=-2)
            return FunctionalModule(pooling_fn)
        else:
            msg = f'Pooling mode {self._pooling_mode} is not supported, possible options are `["mean", "sum"]`'
            raise ValueError(msg)

    def _make_output_layer(
        self,
    ) -> torch.nn.Module:
        """Initializes the output layer."""
        return init_module_from_dict(
            self._output_layers_kwargs, input_dim=self.decoder_input_dim, output_dim=self._output_dim
        )

    def _make_modules(
        self,
    ) -> torch.nn.Module:
        """Initializes the module."""
        layers = {
            **self._make_input_layers(),
            "pooling_layer": self._make_pooling_layer(),
            "output_layer": self._make_output_layer(),
        }
        return torch.nn.ModuleDict(layers)

    def forward(
        self,
        condition_dict: MappedTensor,
    ) -> torch.Tensor:
        """Forward computation pass on the set encoder.

        :param condition_dict: The input dictionary containing the data for
            each perturbation covariate.
        :type condition_dict: class: `MappedTensor`
        """
        # iterating over perturbation covariates
        encoded_covariates = {}
        for covariate_id, covariate_data in condition_dict.items():
            if covariate_id not in self._condition_encoder.keys():
                msg = f"Input encoder not found for covariate {covariate_id}"
                raise KeyError(msg)
            encoded_covariates[covariate_id] = self._condition_encoder[covariate_id](covariate_data)

        pooled_covariates = torch.concatenate(tuple(encoded_covariates.values()), dim=-1)
        pooled_covariates = self._condition_encoder["pooling_layer"](pooled_covariates)
        return self._condition_encoder["output_layer"](pooled_covariates)

    @property
    def decoder_input_dim(
        self,
    ) -> int:
        """Retrieves the input dimensionality for the output decoder."""
        # summing up all the dimensions of non pooled covariates
        decoder_input_dim = sum(layer_dict["output_dim"] for layer_dict in self._input_layers.values())
        return decoder_input_dim
