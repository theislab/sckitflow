from collections.abc import Collection
from typing import Any, Literal

import torch

from sckitflow._types import LayersDict, NestedLayersDict
from sckitflow._utils import check_sequence_query_against_reference
from sckitflow.core._types import MappedTensor
from sckitflow.core.nn._modules import FunctionalModule
from sckitflow.core.nn._utils import init_module_from_dict

__all__ = ["SetEncoder", "TokenAttentionPooling", "SeedAttentionPooling"]


class _AttentionPooling(torch.nn.Module):
    """Pools a set ``[..., n, input_dim]`` with one learned query attending over it."""

    def __init__(
        self, input_dim: int, *, query_dim: int, qkv_dim: int, num_heads: int, dropout_rate: float, token: bool
    ) -> None:
        super().__init__()
        if qkv_dim % num_heads:
            raise ValueError(f"`qkv_dim` ({qkv_dim}) must be divisible by `num_heads` ({num_heads}).")
        self._num_heads = num_heads
        self._dropout_rate = dropout_rate
        self._token = token
        self.query = torch.nn.Parameter(torch.randn(1, 1, query_dim) * query_dim**-0.5)
        self.q = torch.nn.Linear(query_dim, qkv_dim)
        self.k = torch.nn.Linear(input_dim, qkv_dim)
        self.v = torch.nn.Linear(input_dim, qkv_dim)

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        return x.unflatten(-1, (self._num_heads, -1)).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lead = x.shape[:-2]
        x = x.reshape(-1, *x.shape[-2:])
        query = self.query.expand(x.shape[0], -1, -1)
        if self._token:
            x = torch.concatenate([query, x], dim=1)  # the token is part of the set it attends over
        out = torch.nn.functional.scaled_dot_product_attention(
            self._heads(self.q(query)),
            self._heads(self.k(x)),
            self._heads(self.v(x)),
            dropout_p=self._dropout_rate if self.training else 0.0,
        )
        return out.transpose(1, 2).flatten(-2)[:, 0].reshape(*lead, -1)


class TokenAttentionPooling(_AttentionPooling):
    """Prepends a learned token to the set and keeps what it attends to, as CellFlow's ``attention_token``.

    :param input_dim: Dimensionality of the set elements, and of the output.
    :param num_heads: Number of attention heads.
    :param qkv_dim: Dimensionality of queries, keys and values, split across the heads.
    :param dropout_rate: Dropout on the attention weights while training.
    """

    def __init__(self, input_dim: int, num_heads: int = 8, qkv_dim: int = 64, dropout_rate: float = 0.0) -> None:
        super().__init__(
            input_dim, query_dim=input_dim, qkv_dim=qkv_dim, num_heads=num_heads, dropout_rate=dropout_rate, token=True
        )
        self.out = torch.nn.Linear(qkv_dim, input_dim)
        self.output_dim = input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(super().forward(x))


class SeedAttentionPooling(_AttentionPooling):
    """Pooling by multi-head attention with a learned seed (Set Transformer's PMA), as CellFlow's ``attention_seed``.

    :param input_dim: Dimensionality of the set elements.
    :param num_heads: Number of attention heads.
    :param v_dim: Dimensionality of queries, keys and values, split across the heads, and of the output.
    :param seed_dim: Dimensionality of the learned seed.
    :param dropout_rate: Dropout on the attention weights while training.
    """

    # ponytail: no `transformer_block` / `layer_norm` (off by default in CellFlow); add when a model needs them.
    def __init__(
        self, input_dim: int, num_heads: int = 8, v_dim: int = 64, seed_dim: int = 64, dropout_rate: float = 0.0
    ) -> None:
        super().__init__(
            input_dim, query_dim=seed_dim, qkv_dim=v_dim, num_heads=num_heads, dropout_rate=dropout_rate, token=False
        )
        self.output_dim = v_dim


class SetEncoder(torch.nn.Module):
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
        :type pooling_mode: class: `Literal["mean", "sum", "attention-token", "attention-seed"]`

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
            defaults to `None`. Each holds one entry per observation (a set of size 1), e.g. a cell line.
            For ordered slots such as a first and a second drug, give each slot its own condition level.
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
        self._pooling_proj_bias = pooling_proj_bias
        self._covariates_not_pooled = [] if covariates_not_pooled is None else covariates_not_pooled
        self._output_layers_kwargs = {} if output_layers_kwargs is None else output_layers_kwargs
        self._pooling_proj_dim = pooling_proj_dim if pooling_proj_dim else self._min_pooled_dims

        input_layers_dict = self._make_input_layers()
        input_layers = torch.nn.ModuleDict(input_layers_dict)

        # make projection layers
        proj_layers_dict = self._make_proj_layers()
        proj_layers = torch.nn.ModuleDict(proj_layers_dict)

        pooling_layer = self._make_pooling_layer()
        # attention-seed pools to `v_dim`; every other mode keeps the projection dim
        self._pooled_dim = getattr(pooling_layer, "output_dim", self._pooling_proj_dim)
        layers = {
            "input_layers": input_layers,
            "proj_layers": proj_layers,
            "pooling_layer": pooling_layer,
            "output_layer": self._make_output_layer(),
        }
        self._condition_encoder = torch.nn.ModuleDict(layers)

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
            return TokenAttentionPooling(self._pooling_proj_dim, **self._pooling_kwargs)
        elif self._pooling_mode == "attention-seed":
            return SeedAttentionPooling(self._pooling_proj_dim, **self._pooling_kwargs)
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
            if covariate_id not in self._condition_encoder["input_layers"].keys():
                msg = f"Input encoder not found for covariate {covariate_id}"
                raise KeyError(msg)

            # get covariate latent representation
            cov_enc = self._condition_encoder["input_layers"][covariate_id]
            z_cov = cov_enc(covariate_data)

            # update dictionaries
            if covariate_id in self._covariates_not_pooled:
                # categorical covariates carry a set axis `[B, n, D]`; continuous ones are `[B, D]` already
                if z_cov.ndim == 3:
                    if z_cov.shape[-2] != 1:
                        msg = (
                            f"Covariate {covariate_id!r} is not pooled, so it must hold one entry per observation; "
                            f"found {z_cov.shape[-2]}. Pool it, or split its columns into one condition level each."
                        )
                        raise ValueError(msg)
                    z_cov = z_cov.squeeze(-2)
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

        # get the pooled dim if pooled covariates are present
        pooled_dim = self._pooled_dim if len(self.covariates_pooled) > 0 else 0

        # construct decoder input dim
        decoder_input_dim = pooled_dim + sum(not_pooled_input_dims)
        return decoder_input_dim

    @property
    def covariates_pooled(self) -> list[str]:
        """Returns the list of covariates that need to be pooled together."""
        return [cov for cov in self._input_layers.keys() if cov not in self._covariates_not_pooled]
