"""Portable configs for the neural modules.

A module config holds the architecture only; the sizes the data fixes (state and condition dims) are
filled in by :meth:`ModuleConfig.build` from the data module's :class:`~sckitflow.data.DataDimensions`.
"""

from __future__ import annotations

from typing import Any, Literal

import torch
from pydantic import field_validator
from scfit.registry import Component, component

from sckitflow._types import ConditioningLayersId, TimeFeaturesId
from sckitflow.core.nn._vf import MLPVelocity
from sckitflow.data._dims import DataDimensions

__all__ = ["ModuleConfig", "MLPVelocityConfig"]


class ModuleConfig(Component):
    """Family base for anything that configures a neural module."""

    def build(self, dims: DataDimensions) -> torch.nn.Module:
        """The module, sized for ``dims``."""
        raise NotImplementedError


@component("velocity_field.mlp", builds=MLPVelocity)
class MLPVelocityConfig(ModuleConfig):
    """The JSON-able arguments of :class:`MLPVelocity`; see it for their meaning.

    ``state_dim`` and each condition encoder's ``input_dim`` come from the data. The callable
    arguments (``time_features_fn``, ``conditioning_fn``) have no spec, build the module by hand for those.
    """

    encode_state: bool = True
    encode_time: bool = True
    time_features_id: TimeFeaturesId | None = None
    num_time_features: int | None = None
    max_period: int | None = None
    state_encoder_output_dim: int | None = None
    time_encoder_output_dim: int | None = None
    state_encoder_mlp_kwargs: dict[str, Any] | None = None
    time_encoder_mlp_kwargs: dict[str, Any] | None = None
    vf_decoder_mlp_kwargs: dict[str, Any] | None = None
    conditioning_id: ConditioningLayersId | None = None
    conditioning_kwargs: dict[str, Any] | None = None
    condition_encoder_input_layers: dict[str, dict[str, Any]] | None = None
    """One layers dict per condition and group key, without ``input_dim``: the data sets it."""
    condition_encoder_output_dim: int | None = None
    condition_encoder_pooling_mode: Literal["mean", "sum", "attention-token", "attention-seed"] = "mean"
    condition_encoder_pooling_kwargs: dict[str, Any] | None = None
    condition_encoder_pooling_proj_dim: int | None = None
    condition_encoder_pooling_proj_bias: bool = True
    condition_encoder_covariates_not_pooled: tuple[str, ...] | None = None
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
        kwargs = {name: getattr(self, name) for name in type(self).model_fields}
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
        return MLPVelocity(dims.state_dim, **kwargs)
