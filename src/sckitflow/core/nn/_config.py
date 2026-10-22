"""Portable configs for the neural modules.

A module config holds the architecture only; the sizes the data fixes (state and condition dims) are
filled in by :meth:`ModuleConfig.build` from the data module's :class:`~sckitflow.data.DataDimensions`.
"""

from __future__ import annotations

from typing import Any, Literal

import torch
from pydantic import field_validator
from scfit.registry import Component, component

from sckitflow._types import ActivationId, ConditioningLayersId, TimeFeaturesId
from sckitflow.core.nn._modules import MLP
from sckitflow.core.nn._vf import MLPVelocity
from sckitflow.data._dims import DataDimensions

__all__ = ["ModuleConfig", "MLPConfig", "MLPVelocityConfig"]


_ACTIVATIONS: dict[ActivationId, type[torch.nn.Module]] = {
    "relu": torch.nn.ReLU,
    "leaky_relu": torch.nn.LeakyReLU,
    "elu": torch.nn.ELU,
    "gelu": torch.nn.GELU,
    "silu": torch.nn.SiLU,
    "tanh": torch.nn.Tanh,
    "sigmoid": torch.nn.Sigmoid,
    "softplus": torch.nn.Softplus,
}


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------


class ModuleConfig(Component):
    """Family base for anything that configures a neural module."""

    def build(self, dims: DataDimensions) -> torch.nn.Module:
        """The module, sized for ``dims``."""
        raise NotImplementedError


@component("mlp", builds=MLP)
class MLPConfig(ModuleConfig):
    hidden_dims: tuple[int, ...] | None = None
    activation_cls: ActivationId | None = None
    final_activation_cls: ActivationId | None = None
    use_batchnorm: bool = False
    batchnorm_eps: float = 1e-3
    batchnorm_momentum: float = 1e-2
    batchnorm_affine: bool = True
    batchnorm_track_running_stats: bool = True
    use_layernorm: bool = False
    layernorm_eps: float = 1e-5
    layernorm_elementwise_affine: bool = False
    layernorm_bias: bool = False
    dropout_p: float = 0.0
    dropout_inplace: bool = False
    activation_cls_kwargs: dict[str, Any] | None = None
    final_activation_cls_kwargs: dict[str, Any] | None = None
    bias: bool = True

    def mlp_kwargs(self) -> dict[str, Any]:
        """The keyword arguments of :class:`MLP`, with activation names mapped to their classes."""
        kwargs = dict(self)
        for key in ("activation_cls", "final_activation_cls"):
            if kwargs[key] is not None:
                kwargs[key] = _ACTIVATIONS[kwargs[key]]
        return kwargs

    def build(self, input_dim: int, output_dim: int) -> MLP:
        return MLP(input_dim, output_dim, **self.mlp_kwargs())


def _resolve_layer_dict(layers: dict[str, Any]) -> dict[str, Any]:
    """Resolve activation names → classes, recursively.

    Two shapes need resolution:
    * Flat MLP option dicts (``{"activation_cls": "silu", "hidden_dims": …}``)
      for ``condition_encoder_output_layers_kwargs`` / ``source_encoder_mlp_kwargs``.
    * One level of nesting for ``condition_encoder_input_layers``, whose
      values are per-key layers dicts (``{"drug": {"output_dim": 2, …}}``).
    Entries that are already classes (e.g. when pydantic has already run
    ``MLPConfig.mlp_kwargs()``) are left untouched.
    """
    out: dict[str, Any] = {}
    for key, value in layers.items():
        if isinstance(value, dict):
            out[key] = _resolve_layer_dict(value)
        elif key in ("activation_cls", "final_activation_cls") and isinstance(value, str):
            out[key] = _ACTIVATIONS[value]
        else:
            out[key] = value
    return out


def _compile_dict(kwargs_dict: dict[str, Any]) -> dict[str, Any]:
    out_dict: dict[str, Any] = {}
    for arg_name, arg_val in kwargs_dict.items():
        if isinstance(arg_val, MLPConfig):
            out_dict[arg_name] = arg_val.mlp_kwargs()
        elif isinstance(arg_val, dict):
            out_dict[arg_name] = _resolve_layer_dict(arg_val)
        else:
            out_dict[arg_name] = arg_val
    return out_dict


@component("velocity_field.mlp", builds=MLPVelocity)
class MLPVelocityConfig(ModuleConfig):
    """The JSON-able arguments of :class:`MLPVelocity`; see it for their meaning.

    ``state_dim`` and each condition encoder's ``input_dim`` come from the data. The callable
    arguments (``time_features_fn``) have no spec, build the module by hand for those.
    """

    encode_state: bool = True
    encode_time: bool = True
    time_features_id: TimeFeaturesId | None = None
    num_time_features: int | None = None
    max_period: int | None = None
    state_encoder_output_dim: int | None = None
    time_encoder_output_dim: int | None = None
    state_encoder_mlp_kwargs: MLPConfig | None = None
    time_encoder_mlp_kwargs: MLPConfig | None = None
    vf_decoder_mlp_kwargs: MLPConfig | None = None
    conditioning_id: ConditioningLayersId | None = None
    conditioning_kwargs: dict[str, Any] | None = None
    condition_encoder_input_layers: dict[str, dict[str, Any]] | None = None
    """One layers dict per condition and group key, without ``input_dim``: the data sets it."""
    condition_encoder_output_dim: int | None = None
    condition_encoder_pooling_mode: Literal["mean", "sum", "attention-token", "attention-seed"] = "mean"
    condition_encoder_pooling_kwargs: dict[str, Any] | None = None
    condition_encoder_pooling_proj_dim: int | None = None
    condition_encoder_pooling_proj_bias: bool = True
    condition_encoder_output_layers_kwargs: dict[str, Any] | None = None
    source_encoder_mlp_kwargs: dict[str, Any] | None = None
    source_encoder_output_dim: int | None = None

    @field_validator("condition_encoder_input_layers")
    @classmethod
    def _no_input_dim(cls, layers: dict[str, dict[str, Any]] | None) -> dict[str, dict[str, Any]] | None:
        set_by_hand = [key for key, layer in (layers or {}).items() if "input_dim" in layer]
        if set_by_hand:
            raise ValueError(f"`input_dim` comes from the data; drop it from {set_by_hand}.")
        return layers

    def build(self, dims: DataDimensions) -> MLPVelocity:
        kwargs = _compile_dict(dict(self))

        # Fill in per-condition-encoder `input_dim` from the data dimensions.
        if self.condition_encoder_input_layers is not None:
            input_dims = {
                **(dims.condition_reps_dims or {}),
                **(dims.condition_continuous_dims or {}),
                **(dims.groups_reps_dims or {}),
            }
            kwargs["condition_encoder_input_layers"] = {
                key: {**layers, "input_dim": input_dims[key]}
                for key, layers in self.condition_encoder_input_layers.items()
            }
            kwargs["condition_encoder_continuous_covariates"] = list(dims.condition_continuous_dims or ())
        if dims.source_state_dim is not None:
            # The sources have their own state rep: only a source encoder reads it, sized from the data.
            if self.source_encoder_mlp_kwargs is None:
                raise ValueError(
                    "The sources have their own state rep (`source_sample_rep`), which the velocity field only "
                    "reads through a source encoder; without one it would ignore them. Set "
                    "`source_encoder_mlp_kwargs`."
                )
            if "input_dim" in self.source_encoder_mlp_kwargs:
                raise ValueError(
                    "The source encoder's `input_dim` comes from the data; drop it from `source_encoder_mlp_kwargs`."
                )
        return MLPVelocity(dims.state_dim, source_state_dim=dims.source_state_dim, **kwargs)
