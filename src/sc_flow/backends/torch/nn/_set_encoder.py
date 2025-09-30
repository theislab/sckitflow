from collections.abc import Sequence
from typing import Any, Literal

import torch

from sc_flow._types import LayersDict, NestedLayersDict
from sc_flow.backends.torch._types import MappedTensor
from sc_flow.backends.torch._utils import make_concatenation_possible
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
        covariates_not_pooled: Sequence[str] | None = None,
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

        :param covariates_not_pooled: Optional sequence of the perturbation covariate
            identifiers for which the pooling should not be applied. Defaults to `None`.
        :class covariates_not_pooled: class: `Sequence[str] | None`

        :param output_layers_kwargs: Dictionary containing the configurations for the output layer.
            Defaults to `None`.
        :type output_layers_kwargs: class: `LayersDict | None`
        """
        self._input_layers = input_layers
        self._output_dim = output_dim
        self._pooling_mode = pooling_mode
        self._pooling_kwargs = {} if pooling_kwargs is None else pooling_kwargs
        self._covariates_not_pooled = () if covariates_not_pooled is None else covariates_not_pooled
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

    def _get_decoder_input(
        self,
        pooled_covariates: torch.Tensor | None,
        not_pooled_covariates: torch.Tensor | None,
    ) -> torch.Tensor:
        """Retrieves the input tensor to be fed to the decoder.

        At least one of the two input tensor should be not None,
        otherwise a :class: `RuntimeError` is raised. When they are both
        provided, they will be concatenated, otherwise the provided
        one will be returned as is.

        :param pooled_covariates: Tensor containing the data for the
            perturbation covariates that underwent pooling.
        :type pooled_covariates: class: `torch.Tensor`

        :param not_pooled_covariates:Tensor containing the data for the
            perturbation covariates that skipped pooling.
        :type not_pooled_covariates: class: `torch.Tensor`
        """
        if (not_pooled_covariates is not None) and (pooled_covariates is not None):
            return torch.concatenate(
                (pooled_covariates, make_concatenation_possible(not_pooled_covariates, pooled_covariates, -1)), dim=-1
            )
        elif not_pooled_covariates is None:
            return pooled_covariates
        elif pooled_covariates is None:
            return not_pooled_covariates
        else:
            msg = "All covariates are None."
            raise RuntimeError(msg)

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
            # check that the covariate has its encoder defined
            if covariate_id not in self._condition_encoder.keys():
                msg = f"Input encoder not found for covariate {covariate_id}"
                raise KeyError(msg)
            encoded_covariates[covariate_id] = self._condition_encoder[covariate_id](covariate_data)

        # pooled covariates
        if len(self.covariates_to_pool) > 0:
            pooled_covariates = torch.stack(
                [
                    covariate_data
                    for covariate_id, covariate_data in encoded_covariates.items()
                    if covariate_id in self.covariates_to_pool
                ],
                dim=-2,
            )
            pooled_covariates = self._condition_encoder["pooling_layer"](pooled_covariates)
        else:
            pooled_covariates = None

        # not pooled covariates
        if len(self._covariates_not_pooled) > 0:
            not_pooled_covariates = torch.concatenate(
                [
                    covariate_data
                    for covariate_id, covariate_data in encoded_covariates.items()
                    if covariate_id in self._covariates_not_pooled
                ],
                dim=-1,
            )
        else:
            not_pooled_covariates = None

        # handling input for decoder
        decoder_input = self._get_decoder_input(pooled_covariates, not_pooled_covariates)

        # decoder
        return self._condition_encoder["output_layer"](decoder_input)

    @property
    def covariates_to_pool(
        self,
    ) -> Sequence[str]:
        """Retrieves the identifiers of covariate to pool."""
        return [covariate for covariate in self._input_layers.keys() if covariate not in self._covariates_not_pooled]

    @property
    def decoder_input_dim(
        self,
    ) -> int:
        """Retrieves the input dimensionality for the output decoder."""
        # summing up all the dimensions of non pooled covariates
        decoder_input_dim = sum(
            layer_dict["output_dim"]
            for covariate_id, layer_dict in self._input_layers.items()
            if covariate_id in self._covariates_not_pooled
        )

        # retrieving the dimension of the pooled covariates (only one since they are pooled)
        if len(self.covariates_to_pool) > 0:
            decoder_input_dim = decoder_input_dim + self._input_layers[self.covariates_to_pool[0]]["output_dim"]
        return decoder_input_dim
