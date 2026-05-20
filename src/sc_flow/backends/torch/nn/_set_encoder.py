from collections.abc import Collection
from typing import Any, Literal

import torch

from sc_flow._types import LayersDict, NestedLayersDict
from sc_flow._utils import check_sequence_query_against_reference
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
        pooling_mode: Literal["mean", "sum", "attention-token", "attention-seed"] = "mean",
        pooling_kwargs: dict[str, Any] | None = None,
        pooling_proj_dim: int | None = None,
        pooling_proj_bias: bool = True,
        covariates_not_pooled: Collection[str] | None = None,
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

        :param pooling_proj_dim: Shared projection dimension for the covariates to pool,
            defaults to `None`, in which case it will be set to the minimum output
            dimensionality of the input encoder for pooled covariates.
        :type pooling_proj_dim: class: `int | None`

        :param pooling_proj_bias: Whether to use bias term for linear projection of
            covariates to pool, defaults to `True`.
        :type pooling_proj_bias: class: `bool`

        :param covariates_not_pooled: Collection of string identifiers for the covariates not to pool,
            defaults to `None`.
        :type covariates_not_pooled: class: `Collection[str] | None`

        :param output_layers_kwargs: Dictionary containing the configurations for the output layer.
            Defaults to `None`.
        :type output_layers_kwargs: class: `LayersDict | None`
        """
        super().__init__()
        self._input_layers = input_layers
        self._output_dim = output_dim
        self._pooling_mode = pooling_mode
        self._pooling_kwargs = {} if pooling_kwargs is None else pooling_kwargs
        self._pooling_proj_dim = pooling_proj_dim if pooling_proj_dim else self._min_pooled_dims
        self._pooling_proj_bias = pooling_proj_bias
        self._covariates_not_pooled = [] if covariates_not_pooled is None else covariates_not_pooled
        self._output_layers_kwargs = {} if output_layers_kwargs is None else output_layers_kwargs

        self._condition_encoder = self._make_modules()

    @property
    def _min_pooled_dims(self) -> int | None:
        if len(self.covariates_pooled) == 0:
            return None
        dims = [self._input_layers[cov]["output_dim"] for cov in self.covariates_pooled]
        return min(dims)

    def _make_input_layers(
        self,
    ) -> dict[str, torch.nn.Module]:
        """Initializes the input layers."""
        layers = {}
        for covariate_id, covariate_layers_dict in self._input_layers.items():
            layers[covariate_id] = init_module_from_dict(covariate_layers_dict)
        return layers

    def _make_proj_layers(self) -> dict[str, torch.nn.Module]:
        """Initializes the projection layers."""
        layers = {}
        for covariate_id, covariate_layers_dict in self._input_layers.items():
            if covariate_id not in self._covariates_not_pooled:
                # and initialize projection
                cov_out_dim = covariate_layers_dict["output_dim"]
                cov_proj = torch.nn.Linear(
                    cov_out_dim,
                    self._pooling_proj_dim,
                    bias=self._pooling_proj_bias,
                )

                # update dictionary
                layers[covariate_id] = cov_proj
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
        elif self._pooling_mode == "attention-token":
            raise NotImplementedError
        elif self._pooling_mode == "attention-seed":
            raise NotImplementedError
        else:
            msg = f'Pooling mode {self._pooling_mode} is not supported, possible options are `["mean", "sum", "attention-token", "attention-seed"]`'
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
        # make input layers
        input_layers_dict = self._make_input_layers()
        input_layers = torch.nn.ModuleDict(input_layers_dict)

        # make projection layers
        proj_layers_dict = self._make_proj_layers()
        proj_layers = torch.nn.ModuleDict(proj_layers_dict)

        layers = {
            "input_layers": input_layers,
            "proj_layers": proj_layers,
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
        # check that the right keys are present
        check_sequence_query_against_reference(
            condition_dict.keys(),
            self._condition_encoder["input_layers"].keys(),
            allow_missing_from_reference=False,
            allow_missing_from_query=False,
        )

        # prepare dictionary to store encoded covariates
        encoded_covariates_to_pool = {}
        encoded_covariates_not_pooled = {}

        # iterating over perturbation covariates
        for covariate_id, covariate_data in condition_dict.items():
            # check that the covariate is present in the data
            if covariate_id not in self._condition_encoder.keys():
                msg = f"Input encoder not found for covariate {covariate_id}"
                raise KeyError(msg)

            # get covariate latent representation
            cov_enc = self._condition_encoder["input_layers"][covariate_id]
            z_cov = cov_enc(covariate_data)

            # update dictionaries
            if covariate_id in self._covariates_not_pooled:
                encoded_covariates_not_pooled[covariate_id] = z_cov

            else:
                # get shared projection layer
                cov_proj = self._condition_encoder["proj_layers"][covariate_id]

                # apply projection and update dict
                z_cov = cov_proj(z_cov)
                encoded_covariates_to_pool[covariate_id] = z_cov

        # pooled covariates
        if len(encoded_covariates_to_pool) > 0:
            pooled_covariates = torch.concatenate(tuple(encoded_covariates_to_pool.values()), dim=-2)
            pooled_covariates = self._condition_encoder["pooling_layer"](pooled_covariates)
        else:
            pooled_covariates = None

        # not pooled covariates
        if len(encoded_covariates_not_pooled) > 0:
            covariates_not_pooled = torch.concatenate(tuple(encoded_covariates_not_pooled.values()), dim=-1)
        else:
            covariates_not_pooled = None

        # get joint representation
        to_concat = []
        if pooled_covariates is not None:
            to_concat.append(pooled_covariates)
        if covariates_not_pooled is not None:
            to_concat.append(covariates_not_pooled)
        if len(to_concat) == 0:
            msg = "No condition covariate found."
            raise ValueError(msg)
        latent_cond = torch.concatenate(to_concat, dim=-1)

        return self._condition_encoder["output_layer"](latent_cond)

    @property
    def decoder_input_dim(
        self,
    ) -> int:
        """Retrieves the input dimensionality for the output decoder."""
        # define list to store dimensions
        not_pooled_input_dims = []

        # iterate over each covariate
        for cov, cov_dict in self._input_layers.items():
            # update store
            if cov in self._covariates_not_pooled:
                output_dim = cov_dict["output_dim"]
                not_pooled_input_dims.append(output_dim)

        # get pooling projection dim if pooled covariates are present
        if len(self.covariates_pooled) > 0:
            pooling_proj_dim = self._pooling_proj_dim
        else:
            pooling_proj_dim = 0

        # construct decoder input dim
        decoder_input_dim = pooling_proj_dim + sum(not_pooled_input_dims)
        return decoder_input_dim

    @property
    def covariates_pooled(self) -> list[str]:
        """Returns the list of covariates that need to be pooled together."""
        return [cov for cov in self._input_layers.keys() if cov not in self._covariates_not_pooled]
