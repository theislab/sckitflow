from collections.abc import Collection, Mapping
from typing import Literal

import torch

from sc_flow._types import LayersDict, NestedLayersDict
from sc_flow._utils import check_sequence_query_against_reference
from sc_flow.core._torch_types import MappedTensor
from sc_flow.core.nn._modules import BaseModule
from sc_flow.core.nn._utils import init_module_from_dict
from sc_flow.flow._pooling import BasePooling, PoolingSpec, build_pooling, validate_pooling_spec

__all__ = ["SetEncoder"]


class SetEncoder(BaseModule):
    """Permutation-invariant Deep-Sets encoder over a set of perturbation covariates (the *condition encoder*).

    Torch port of cellflow's ``ConditionEncoder`` (theislab/cellflow, ``src/cellflow/networks/_set_encoders.py``,
    flax) — kept structurally aligned so the jax original and this port stay mutually reviewable. Same shape: a
    per-covariate input layer, a shared projection + pooling, covariates that bypass pooling
    (``covariates_not_pooled``), an output layer, and a deterministic or stochastic (mean + log-variance) head.

    A :class:`~sc_flow.flow._pooling.PoolingSpec` JSON mapping selects a portable built-in pooling implementation. A custom
    :class:`~sc_flow.flow._pooling.BasePooling` instance is an explicitly runtime-only research escape hatch:
    it can train and participate in trusted checkpoints, but :meth:`save_pretrained` refuses portable export.
    """

    def __init__(
        self,
        input_layers: NestedLayersDict,
        output_dim: int,
        pooling: PoolingSpec | BasePooling,
        pooling_proj_dim: int | None = None,
        pooling_proj_bias: bool = True,
        covariates_not_pooled: Collection[str] | None = None,
        output_layers_kwargs: LayersDict | None = None,
        condition_mode: Literal["deterministic", "stochastic"] = "deterministic",
    ) -> None:
        """Initializes the set encoder.

        :param input_layers: Dictionary mapping each perturbation covariate
            identifier to the configurations for their respective input layer.
        :type input_layers: class: `NestedLayersDict`

        :param output_dim: The output dimensionality of the set encoder.
        :type output_dim: class: `int`

        :param pooling: Portable built-in :class:`PoolingSpec` JSON mapping, or a custom
            :class:`BasePooling` instance for runtime experimentation. This choice is required explicitly.

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
        self._pooling_proj_bias = pooling_proj_bias
        self._covariates_not_pooled = tuple(() if covariates_not_pooled is None else covariates_not_pooled)
        self._output_layers_kwargs = {} if output_layers_kwargs is None else output_layers_kwargs
        self._condition_mode = condition_mode

        check_sequence_query_against_reference(
            self._covariates_not_pooled,
            self._input_layers.keys(),
            allow_missing_from_query=True,
            allow_missing_from_reference=False,
        )
        self._pooling_proj_dim = pooling_proj_dim if pooling_proj_dim is not None else self._min_pooled_dims
        if self.covariates_pooled and (self._pooling_proj_dim is None or self._pooling_proj_dim <= 0):
            raise ValueError(f"pooling_proj_dim must be positive when covariates are pooled, found {pooling_proj_dim}.")

        self._custom_pooling = isinstance(pooling, BasePooling)
        if self._custom_pooling:
            if not self.covariates_pooled:
                raise ValueError(
                    "A custom pooling instance was supplied, but no covariates are configured for pooling."
                )
            if pooling.input_dim != self._pooling_proj_dim:
                raise ValueError(
                    f"Custom pooling expects input_dim={pooling.input_dim}, but SetEncoder projects to "
                    f"pooling_proj_dim={self._pooling_proj_dim}."
                )
            # The module is registered exactly once below, inside ``pooling_layer``. A normal assignment
            # here would register the same parameters both as ``_pooling.*`` and under the ModuleDict.
            object.__setattr__(self, "_pooling", pooling)
        else:
            self._pooling = validate_pooling_spec(pooling)

        # PyTorchModelHubMixin captures constructor arguments before __init__. Replace the caller's mapping
        # with the validated copy so the persisted config exactly matches the module that was built.
        if not self._custom_pooling and isinstance(self._hub_mixin_config, dict):
            self._hub_mixin_config["pooling"] = self.pooling_spec
            self._hub_mixin_config["covariates_not_pooled"] = list(self._covariates_not_pooled)

        self._condition_encoder = self._make_modules()

    @property
    def output_dim(self) -> int:
        """The output dimensionality of the condition embedding."""
        return self._output_dim

    @property
    def is_stochastic(self) -> bool:
        """Whether the encoder is variational (outputs a mean **and** a log-variance head)."""
        return self._condition_mode == "stochastic"

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
        if not self.covariates_pooled:
            return torch.nn.Identity()
        if isinstance(self._pooling, BasePooling):
            return self._pooling
        return build_pooling(self._pooling, input_dim=self._pooling_proj_dim)

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
        # A stochastic (VAE-style) encoder gets a second head for the log-variance (same shape as the mean).
        if self.is_stochastic:
            layers["var_layer"] = self._make_output_layer()
        return torch.nn.ModuleDict(layers)

    def forward(
        self,
        condition_dict: MappedTensor,
        condition_mask: Mapping[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward computation pass on the set encoder.

        Returns ``(mean, logvar)`` — the pooled condition embedding (mean) and, for a stochastic
        (VAE-style) encoder, its log-variance head; ``logvar`` is ``None`` when deterministic.

        :param condition_dict: The input dictionary containing the data for
            each perturbation covariate.
        :type condition_dict: class: `MappedTensor`

        :param condition_mask: Optional boolean valid-element mask per pooled covariate, each with shape
            ``(batch, set)``. ``None`` declares a dense, unpadded input and takes a mask-free fast path.
            When supplied, every pooled covariate must have a mask; partial mappings and masks for bypassed
            covariates are rejected.
        """
        if not condition_dict:
            raise ValueError("No condition covariate found.")

        # check that the right keys are present
        check_sequence_query_against_reference(
            condition_dict.keys(),
            self._condition_encoder["input_layers"].keys(),
            allow_missing_from_reference=False,
            allow_missing_from_query=False,
        )
        if condition_mask is not None:
            expected_masks = set(self.covariates_pooled)
            supplied_masks = set(condition_mask)
            if supplied_masks != expected_masks:
                missing = sorted(expected_masks - supplied_masks)
                unexpected = sorted(supplied_masks - expected_masks)
                raise ValueError(
                    "An explicit condition_mask must contain every pooled covariate and no others; "
                    f"missing={missing}, unexpected={unexpected}."
                )

        # prepare dictionary to store encoded covariates
        encoded_covariates_to_pool = {}
        encoded_covariates_not_pooled = {}
        pooled_masks = None if condition_mask is None else []

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
                if z_cov.ndim != 2:
                    raise ValueError(
                        f"Bypassed covariate {covariate_id!r} must encode to (batch, features), "
                        f"found {tuple(z_cov.shape)}."
                    )
                encoded_covariates_not_pooled[covariate_id] = z_cov

            else:
                # get shared projection layer
                cov_proj = self._condition_encoder["proj_layers"][covariate_id]

                # apply projection and update dict
                z_cov = cov_proj(z_cov)
                if z_cov.ndim != 3:
                    raise ValueError(
                        f"Pooled covariate {covariate_id!r} must encode to (batch, set, features), "
                        f"found {tuple(z_cov.shape)}."
                    )
                encoded_covariates_to_pool[covariate_id] = z_cov
                if pooled_masks is not None:
                    mask = condition_mask[covariate_id]
                    if mask.shape != z_cov.shape[:2]:
                        raise ValueError(
                            f"Mask for {covariate_id!r} must have shape {tuple(z_cov.shape[:2])}, "
                            f"found {tuple(mask.shape)}."
                        )
                    pooled_masks.append(mask)

        # pooled covariates
        if len(encoded_covariates_to_pool) > 0:
            pooled_covariates = torch.concatenate(tuple(encoded_covariates_to_pool.values()), dim=-2)
            combined_mask = None if pooled_masks is None else torch.concatenate(pooled_masks, dim=-1)
            pooled_covariates = self._condition_encoder["pooling_layer"](pooled_covariates, combined_mask)
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
        latent_cond = torch.concatenate(to_concat, dim=-1)

        mean = self._condition_encoder["output_layer"](latent_cond)
        logvar = self._condition_encoder["var_layer"](latent_cond) if self.is_stochastic else None
        return mean, logvar

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

        # get the pooling output dim if pooled covariates are present (seed attention can change it)
        if len(self.covariates_pooled) > 0:
            pooling_output_dim = self.pooling_output_dim
        else:
            pooling_output_dim = 0

        # construct decoder input dim
        decoder_input_dim = pooling_output_dim + sum(not_pooled_input_dims)
        return decoder_input_dim

    @property
    def covariates_pooled(self) -> list[str]:
        """Returns the list of covariates that need to be pooled together."""
        return [cov for cov in self._input_layers.keys() if cov not in self._covariates_not_pooled]

    @property
    def pooling_spec(self) -> PoolingSpec | None:
        """Canonical portable pooling spec, or ``None`` for a runtime-only custom instance."""
        if isinstance(self._pooling, BasePooling):
            return None
        return PoolingSpec(
            type=self._pooling["type"],
            version=self._pooling["version"],
            config=dict(self._pooling["config"]),
        )

    @property
    def pooling_output_dim(self) -> int:
        """Dimensionality produced by the configured pooling component."""
        if not self.covariates_pooled:
            return 0
        if isinstance(self._pooling, BasePooling):
            return self._pooling.output_dim
        if self._pooling["type"] == "sc_flow.attention_seed":
            return self._pooling["config"]["v_dim"]
        return self._pooling_proj_dim
