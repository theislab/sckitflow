from __future__ import annotations

from collections.abc import Mapping

import torch
from scfit.registry import Component, component

from sckitflow.core._types import MappedTensor
from sckitflow.core._utils import make_concatenation_possible
from sckitflow.core.nn.blocks._mlp import MLPConfig, MLPEncoderConfig
from sckitflow.core.nn.blocks._pooling import MeanPoolingConfig, PoolingLayerConfig

__all__ = ["SetEncoderConfig", "SetEncoder"]


# All the fields that will be registered to the set encoder's neural networks.
# The keys ``_CONTINUOUS_LAYERS_KEY`` and ``_CATEGORICAL_LAYERS_KEY`` are nested inside
# ``_INPUT_LAYERS_KEY``
_INPUT_LAYERS_KEY = "input_layers"
_CONTINUOUS_LAYERS_KEY = "continuous"
_CATEGORICAL_LAYERS_KEY = "categorical"
_PROJECTION_LAYERS_KEY = "projection"
_POOLING_LAYER_KEY = "pooling"
_OUTPUT_LAYER_KEY = "output_layer"


class SetEncoder(torch.nn.Module):
    """Encoder for set of conditioning covariates."""

    def __init__(
        self,
        cfg: SetEncoderConfig,
        *,
        categorical_dims: Mapping[str, int],
        continuous_dims: Mapping[str, int],
    ) -> None:
        """Initializes the set encoder from the given configurations and per-covariate input dimensionalities."""
        # The first thing we have to do in any case, is to call the constructor of the parent class.
        super().__init__()

        # ------------------------------------------------------------------------------------------
        # ---- A. Verify and register attributes + set defaults. -----------------------------------
        # ------------------------------------------------------------------------------------------
        # Initialize defaults.
        categorical_input_layers = {} if cfg.categorical_input_layers is None else cfg.categorical_input_layers
        continuous_input_layers = {} if cfg.continuous_input_layers is None else cfg.continuous_input_layers

        pooling_cfg = MeanPoolingConfig() if cfg.pooling is None else cfg.pooling
        output_layer_cfg = MLPConfig() if cfg.output_layer is None else cfg.output_layer

        # Fail early when we dont have neither continuous nor categorical input layers.
        if len(categorical_input_layers) == 0 and len(continuous_input_layers) == 0:
            raise ValueError(
                "Set encoder needs one of ``categorical_input_layers`` of ``continuous_input_layers``"
                "to be initialized, but none was found."
            )

        # A categorical covariate is one row per batch (the batch's group), so batch statistics over it are
        # meaningless; so are the output layer's when nothing per observation reaches it.
        normed = [k for k, v in categorical_input_layers.items() if v.mlp.use_batchnorm]
        if len(continuous_input_layers) == 0 and output_layer_cfg.use_batchnorm:
            normed.append(_OUTPUT_LAYER_KEY)
        if len(normed) > 0:
            raise ValueError(
                f"`use_batchnorm` is set on {normed}, but these layers only ever see one row per batch: a batch "
                "is one group, so a categorical covariate's encoding is the same for all of its observations and "
                "is passed as a single row. Batch statistics over one row are undefined. Turn batch "
                "normalization off for these layers (`use_layernorm` works on a single row)."
            )

        # ------------------------------------------------------------------------------------------
        # ---- B. Initialize neural networks. ------------------------------------------------------
        # ------------------------------------------------------------------------------------------
        # Initialize store for the entire set encoder. This will be the outermost dictionary that we
        # will use to initialize the set encoder.
        set_encoder_dict = {}
        # ---- B-1. Build the input layers. ----
        # Categorical and continuous covariate encoders will be stored in two different ``toch.nn.ModuleDict``s.
        input_layers_dict = {}
        # We pack together a string descriptor of the covariate type, the input layers and the
        # corresponding data dimensionalities -- we iterate over this packed tuples and validate
        # them against the data dimensionalities.
        for input_type, type_specific_layers, type_specific_dims in [
            (_CONTINUOUS_LAYERS_KEY, continuous_input_layers, continuous_dims),
            (_CATEGORICAL_LAYERS_KEY, categorical_input_layers, categorical_dims),
        ]:
            # We create two separate ``torch.nn.ModuleDict`` for categorical and continuous
            # condition covariates. Here, we initialize the dictoinary to store the type-specific
            # ``ModuleDict``, which are then nested in a single dictionary for all the input layers.
            type_specific_layers_dict = {}
            # For each input type, we then iterate over the individual covariates.
            # We firts verify the covariate against the data dimensionalities, then
            # build the encoder and finally register it to ``input_layers_dict``
            for covariate, covariate_cfg in type_specific_layers.items():
                # Perform the validation against the data dimensionality, raise otherwise.
                if covariate not in type_specific_dims:
                    raise ValueError(
                        f"Failed to initialize {input_type} covariate {covariate};"
                        "no corresponding data dimensionality was found. "
                        f"The available keys are {sorted(type_specific_dims)}."
                    )
                # Now we can finally initialize and register the encoder block.
                type_specific_layers_dict[covariate] = covariate_cfg.build(type_specific_dims[covariate])
            # We update the input layers with the corresponding ``torch.nn.ModuleDict``.
            input_layers_dict[input_type] = torch.nn.ModuleDict(type_specific_layers_dict)

        # All the ``input_layers`` will constitute a single ``torch.nn.ModuleDict``.
        # The layers for categorical and continuous covariates will however be split and nested inside
        # this single ``input_layers`` dictionary. We register this module to the main dictionary.
        set_encoder_dict[_INPUT_LAYERS_KEY] = torch.nn.ModuleDict(input_layers_dict)

        # ---- B-2. Build the pooling projection layers. ----
        # This linear projection is required to
        # map all the categorical covariates over a comparable space for concatenation.
        # Given the current architecture, no projection is needed for continuous covariates.
        # TODO: if we decide to pool continuous covariates as well, they will also need to
        # be mapped though this linear projector. For the moment, this is not the case --
        # the current implementation concatenates the continuous covariates, instead of
        # including them in the pooling step.
        # NOTE/TODO: It might be interesting to consider the option of sharing the projection layer
        # across all the covariates. This would constrain the model to learn a single subspace over which to map
        # all the categorical covariates -- and maybe the continuous covariates as well, if we decide to add them to the pooling step.
        if len(categorical_input_layers) > 0:
            # We retrieve the pooling projection dimension
            if cfg.pooling_proj_dim is None:
                pooling_proj_dim = min(covariate_cfg.output_dim for covariate_cfg in categorical_input_layers.values())
            else:
                pooling_proj_dim = cfg.pooling_proj_dim

            set_encoder_dict[_PROJECTION_LAYERS_KEY] = torch.nn.ModuleDict(
                {
                    covariate: torch.nn.Linear(
                        covariate_cfg.output_dim,
                        pooling_proj_dim,
                        bias=cfg.pooling_proj_bias,
                    )
                    for covariate, covariate_cfg in categorical_input_layers.items()
                }
            )

            # This layer will be initialized using ``pooling_proj_dim``,
            # because all the covariates will be linearly projected to the same space. They will be concatenated
            # over their second-to-last dimension, and finally pooled to collapse the combinatorial dimension.
            # TODO: If we decide to pool the continuous covariates too, we need to pass them through the linear
            # projector and the pooling layer.
            set_encoder_dict[_POOLING_LAYER_KEY] = pooling_cfg.build(pooling_proj_dim)

            # We compute here the contribution of categorical covariates to the decoder input dimension. This will
            # be done by the trailing dimension of the pooling layer's output. We guard with the length of
            # ``categorical_input_layers`` because this contribution should only be considered when the
            # set encoder models categorical covariates. The ``categorical_out_dim`` variable will hold such component.
            # Attention-seed pools to `v_dim`; every other mode keeps the projection dim.
            # We rely on the public API of pooling layers, that are required to expose their output dimension.
            categorical_out_dim = set_encoder_dict[_POOLING_LAYER_KEY].output_dim

        # When there are no categorical covariates, the only contribution comes from the continuous covariates;
        # therefore, we set ``pooled_dim`` to ``0``.
        else:
            categorical_out_dim = 0

        # ---- B-3. Build the output layer. ----
        # Computing the component coming from the continuous covariates.
        continuous_out_dim = sum(covariate_cfg.output_dim for covariate_cfg in continuous_input_layers.values())

        # Finally -- once we have computed both categorical and continuous covariates latent dimensionalities --
        # we can build the output layer.
        set_encoder_dict[_OUTPUT_LAYER_KEY] = output_layer_cfg.build(
            categorical_out_dim + continuous_out_dim, cfg.output_dim
        )

        # We ultimately register all the neural networks in ``set_encoder_dict`` in a single ``torch.nn.ModuleDict``,
        # which will be the main and only entrypoint for the forward pass.
        self._set_encoder = torch.nn.ModuleDict(set_encoder_dict)

    def forward(
        self,
        condition_dict: MappedTensor,
    ) -> torch.Tensor:
        """Forward computation pass on the set encoder.

        :param condition_dict: The input dictionary containing the data for
            each perturbation covariate.
        :type condition_dict: class: `MappedTensor`
        """
        # ---- Encode continuous covariates ----
        continuous_covariates_dict = {}
        for covariate, covariate_layer in self._set_encoder[_INPUT_LAYERS_KEY][_CONTINUOUS_LAYERS_KEY].items():
            if covariate not in condition_dict:
                raise ValueError(
                    f"Continuous covariate {covariate} not found in {condition_dict}. "
                    f"Available keys are {sorted(condition_dict)}."
                )
            x_covariate = condition_dict.get(covariate)
            continuous_covariates_dict[covariate] = covariate_layer(x_covariate)

        # ---- Encode categorical covariates ----
        categorical_covariates_dict = {}
        for covariate, covariate_layer in self._set_encoder[_INPUT_LAYERS_KEY][_CATEGORICAL_LAYERS_KEY].items():
            if covariate not in condition_dict:
                raise ValueError(
                    f"Categorical covariate {covariate} not found in {condition_dict}. "
                    f"Available keys are {sorted(condition_dict)}."
                )
            x_covariate = condition_dict.get(covariate)
            z_covariate = covariate_layer(x_covariate)
            categorical_covariates_dict[covariate] = self._set_encoder[_PROJECTION_LAYERS_KEY][covariate](z_covariate)

        # ---- Pool categorical covariates ----
        if len(categorical_covariates_dict):
            categorical_covariates = torch.concatenate(tuple(categorical_covariates_dict.values()), dim=-2)
            categorical_covariates = self._set_encoder[_POOLING_LAYER_KEY](categorical_covariates)
        else:
            categorical_covariates = None

        # ---- Concatenate all the continuous covariates ---
        if len(continuous_covariates_dict) > 0:
            continuous_covariates = torch.concatenate(tuple(continuous_covariates_dict.values()), dim=-1)
        else:
            continuous_covariates = None

        # ---- Forward pass on output layer ----
        # We raise an error if no covariate was found. We make this check explicit, even though it
        # is guaranteed from the validation performed at initialization, and the one perfomed at the
        # beginning of the forward pass.
        if categorical_covariates is None and continuous_covariates is None:
            raise ValueError("Neither continuous nor categorical covariates were found.")
        # No categorical covariates; the output layer processes only the continuous ones.
        if categorical_covariates is None:
            return self._set_encoder[_OUTPUT_LAYER_KEY](continuous_covariates)
        # No continuous covariates; the output layer processes only the categorical ones.
        elif continuous_covariates is None:
            return self._set_encoder[_OUTPUT_LAYER_KEY](categorical_covariates)
        # Both continuous and categorical covariates present; we concatenate and pass them to the ouput layer.
        else:
            categorical_covariates = make_concatenation_possible(categorical_covariates, continuous_covariates, -1)
            latent_cond = torch.concatenate((categorical_covariates, continuous_covariates), dim=-1)
            return self._set_encoder[_OUTPUT_LAYER_KEY](latent_cond)


@component("nn.composite.set_encoder", builds=SetEncoder)
class SetEncoderConfig(Component):
    """Configuration class for set encoders."""

    output_dim: int
    """
    :param output_dim: The output dimensionality of the set encoder.
    :type output_dim: `int`
    """

    categorical_input_layers: dict[str, MLPEncoderConfig] | None = None
    """
    :param categorical_input_layers: Dictionary of ``MLPEncoderConfig``, mapping each categorical
        condition covariate to the configuration for the corresponding feature extractor.
    :type categorical_input_layers: `dict[str, MLPEncoderConfig]`
    """

    continuous_input_layers: dict[str, MLPEncoderConfig] | None = None
    """
    :param continuous_input_layers: Dictionary of ``MLPEncoderConfig``, mapping each continuous
        condition covariate to the configuration for the corresponding feature extractor.
    :type continuous_input_layers: `dict[str, MLPEncoderConfig]`
    """

    pooling: PoolingLayerConfig | None = None
    """
    :param pooling: Optional configuration for the pooling layer. When `None`, it will be
        automatically initialized to `MeanPoolingConfig`.
    :type pooling: `PoolingLayerConfig | None`
    """

    pooling_proj_dim: int | None = None
    """
    :param pooling_proj_dim: Shared projection dimension for the covariates to pool,
        defaults to `None`, in which case it will be set to the minimum output
        dimensionality of the input encoder for pooled covariates.
    :type pooling_proj_dim: class: `int | None`
    """

    pooling_proj_bias: bool = True
    """
    :param pooling_proj_bias: Whether to use bias term for linear projection of
        covariates to pool, defaults to `True`.
    :type pooling_proj_bias: class: `bool`
    """

    output_layer: MLPConfig | None = None
    """
    :param output_layer: The configurations for the output MLP layer.
    :type output_layer: `MLPConfig | None`
    """

    def build(self, categorical_dims: Mapping[str, int], continuous_dims: Mapping[str, int]) -> SetEncoder:
        return SetEncoder(self, categorical_dims=categorical_dims, continuous_dims=continuous_dims)
