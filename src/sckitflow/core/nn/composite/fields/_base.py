from __future__ import annotations

import abc
from collections.abc import Callable
from functools import partial
from typing import Protocol

import torch
from scfit.registry import component

from sckitflow.core._utils import ensure_2d_tensor_with_singleton_trailing_dim, make_concatenation_possible
from sckitflow.core.nn._config import ModuleConfig
from sckitflow.core.nn.blocks._mlp import MLPConfig, MLPEncoderConfig
from sckitflow.core.nn.blocks._time_features import TimeFeaturesConfig
from sckitflow.core.nn.blocks.conditioning._base import BaseConditioningLayerConfig
from sckitflow.core.nn.blocks.conditioning._concat import ConcatConditioningConfig
from sckitflow.core.nn.composite._set_encoder import SetEncoderConfig
from sckitflow.data._dims import DataDimensions

__all__ = ["Field", "NeuralVelocity", "NeuralVelocityConfig"]


class Field(Protocol):
    def forward(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        *,
        condition_dict: dict[str, torch.Tensor] | None = None,
        source: torch.Tensor | None = None,
    ) -> torch.Tensor: ...


_STATE_ENCODER_KEY = "state_encoder"
_CONDITIONING_LAYER_KEY = "conditioning"
_CONDITION_ENCODER_KEY = "condition_encoder"
_SOURCE_ENCODER_KEY = "source_encoder"
_TIME_FEATURES_KEY = "time_features"
_T_ENCODER_KEY = "t_encoder"
_FIELD_DECODER_KEY = "field_decoder"


class _NeuralFieldMixin(abc.ABC, torch.nn.Module):
    def __init__(
        self,
        cfg: _BaseNeuralFieldConfig,
        *,
        dims: DataDimensions,
    ) -> None:
        # The first thing we have to do in any case, is to call the constructor of the parent class.
        super().__init__()

        # ------------------------------------------------------------------------------------------
        # ---- A. Verify and register attributes + set defaults. -----------------------------------
        # ------------------------------------------------------------------------------------------
        # Sources in their own state space (``dims.source_state_dim`` set) can only be read through a source
        # encoder. Sources sharing the targets' state (``None``) may still be encoded, from ``dims.state_dim``.
        if cfg.source_encoder is None and dims.source_state_dim is not None:
            raise ValueError(
                "The data configurations imply source conditioning; please pass the "
                "``source_encoder`` field to the configuration (modeling source conditioning) "
                "demands the presence of a source encoder."
            )

        # Regardless of the configurations, we are always going to require a conditioning layer
        # and a field decoder. We initialize these to their defaults when they are not provided.
        conditioning_cfg = ConcatConditioningConfig() if cfg.conditioning is None else cfg.conditioning
        field_decoder_cfg = MLPConfig() if cfg.field_decoder is None else cfg.field_decoder

        # ------------------------------------------------------------------------------------------
        # ---- B. Initialize neural networks. ------------------------------------------------------
        # ------------------------------------------------------------------------------------------
        # Initialize store for the entire field. This will be the outermost dictionary that we
        # will use to initialize the neural fields.
        field_dict = {}
        # ---- B-1 Build state encoder ----
        # When builiding the state decoder, we need to store its output dimensions; this will be
        # needed to compute the input dimension used to initialize the field decoder.
        if cfg.state_encoder is not None:
            latent_state_dim = cfg.state_encoder.output_dim
            field_dict[_STATE_ENCODER_KEY] = cfg.state_encoder.build(dims.state_dim)
        # When no state encoder is provided, the latent state dimensionality will remain unchanged.
        # However, we still need to store it under the same name.
        else:
            latent_state_dim = dims.state_dim

        # ---- B-2 Build time features and time encoder ----
        # NOTE: This is the problematic piece of logic to share between velocity and flow map fields,
        # as the latter will need to encode two time indices instead of only one. This difference also
        # affects the construction of the conditioning layer, as the contribution of the time
        # information to the ``latent_time_dim`` will be doubled.

        # Same thing for the time features: we need to register the resulting dimensionality (depending
        # on the provided configuration) to carry the desired dimensionality forward until the (optional)
        # initialization of the time decoder; such dimensionality will be transitively carried through
        # both the optional time features and encoder.
        # Build the time features: they will only be there when explicitly provided, otherwise they
        # will be modeled as a scalar index.
        if cfg.time_features is not None:
            n_time_features = cfg.time_features.n_time_features
            field_dict[_TIME_FEATURES_KEY] = cfg.time_features.build()
        else:
            n_time_features = 1
        # We then build the time encoder, taking as input the number of time features, resolved
        # by the presence of the corresponding time-features configurations.
        if cfg.time_encoder is not None:
            latent_time_dim = cfg.time_encoder.output_dim
            field_dict[_T_ENCODER_KEY] = cfg.time_encoder.build(n_time_features)
        else:
            latent_time_dim = n_time_features

        # ---- B-4 Build condition encoder ----
        # The field entails two optional sources of conditioning: the actual conditions and the source state.
        # To determine the latent dimensionality of the conditioning signal, we need the two terms up.
        latent_condition_dims_to_sum = []
        # We build the optional condition encoder; condition modeling is only allowed when the configurations
        # for the set encoder are provided. An unconditional field will be instead modeled otherwise.
        if cfg.condition_encoder is not None:
            # Groups reach the condition dict as categorical encodings, next to the categorical conditions.
            field_dict[_CONDITION_ENCODER_KEY] = cfg.condition_encoder.build(
                categorical_dims={**(dims.groups_reps_dims or {}), **(dims.condition_reps_dims or {})},
                continuous_dims=dims.condition_continuous_dims or {},
            )
            # We update the list of conditioning signals' dimensionality with the condition encoder's output dimension.
            latent_condition_dims_to_sum.append(cfg.condition_encoder.output_dim)

        # ---- B-5 Build source encoder ----
        # We build the optional source encoder -- this module is needed to enforce source-conditioning, required
        # in case-control studies when the interpolation occurs between noise and samples from the target distribution.
        # Source-conditioning always require the definition of the source encoder, and an error is thrown otherwise.
        if cfg.source_encoder is not None:
            field_dict[_SOURCE_ENCODER_KEY] = cfg.source_encoder.build(dims.source_state_dim or dims.state_dim)
            # We update the list of conditioning signals' dimensionality with the source encoder's output dimension.
            latent_condition_dims_to_sum.append(cfg.source_encoder.output_dim)

        # ---- B-6 Build conditioning layer -----
        # To build the conditioning layer, we first need to evaluate the latent dimensionality of the conditioning
        # signal. We only pass it when there is at least one conditioning signal, and we set it to ``None`` otherwise.
        latent_condition_dim = sum(latent_condition_dims_to_sum) or None
        # NOTE: We may need to adapt this building step to accommodate flow map fields: the only required change would
        # be to double the value of ``latent_dim_dim`` before initializing the conditioning layer. One possible
        # way to achieve this, would be to expose the ``h_time_dim`` property from the ``NeuralFieldConfig``, and
        # override it for flow map configurations.
        # Then, we initialize the conditioning layer, which will always be required regardless of the configurations.
        field_dict[_CONDITIONING_LAYER_KEY] = conditioning_cfg.build(
            latent_state_dim=latent_state_dim,
            latent_time_dim=latent_time_dim,
            latent_condition_dim=latent_condition_dim,
        )

        # ---- B-7 Build field decoder ----
        # Similarly to the conditioning layer, the field decoder will always be present.
        field_dict[_FIELD_DECODER_KEY] = field_decoder_cfg.build(
            field_dict[_CONDITIONING_LAYER_KEY].output_dim, dims.state_dim
        )
        # We then register all these neural modules as a single ``torch.nn.ModuleDict``.
        self._field = torch.nn.ModuleDict(field_dict)

    def _encode_time(self, t: torch.Tensor) -> torch.Tensor:
        # We firts retrieve the optional time feature extractor. When available, we transform
        # the time index with it; otherwise, we keep it unchanged.
        if _TIME_FEATURES_KEY in self._field:
            t = self._field[_TIME_FEATURES_KEY](t)
        else:
            # raw time comes as ``()`` (ODE solvers) or ``(B,)`` (training); the layers want a trailing time axis
            t = ensure_2d_tensor_with_singleton_trailing_dim(t)

        if _T_ENCODER_KEY in self._field:
            return self._field[_T_ENCODER_KEY](t)
        return t

    def _encode_state(self, x: torch.Tensor) -> torch.Tensor:
        if _STATE_ENCODER_KEY in self._field:
            return self._field[_STATE_ENCODER_KEY](x)
        return x

    def _encode_condition(
        self, cond_dict: dict[str, torch.Tensor] | None = None, source: torch.Tensor | None = None
    ) -> torch.Tensor | None:
        # Check from the data side: at least one of the two conditioning
        # signals must be passed for this function to be correctly called.
        # If this is not the case, we return ``None``.
        if cond_dict is None and source is None:
            return None
        # Same check, but from the network side: we need at least one of source
        # or condition encoder to be present.
        # If this is not the case, we return ``None``.
        if (_CONDITION_ENCODER_KEY not in self._field) and (_SOURCE_ENCODER_KEY not in self._field):
            return None

        # Encode condition data.
        if _CONDITION_ENCODER_KEY in self._field:
            if cond_dict is None:
                raise ValueError(
                    "The field was initialized with a condition encoder, but no condition data was passed to the model."
                )
            z_cond = self._field[_CONDITION_ENCODER_KEY](cond_dict)
        else:
            z_cond = None

        # Encode source state data.
        if _SOURCE_ENCODER_KEY in self._field:
            if source is None:
                raise ValueError(
                    "The field was initialized with a source encoder, but no source state was passed to the model."
                )
            z_source = self._field[_SOURCE_ENCODER_KEY](source)
        else:
            z_source = None

        # First, we need to explicity check that both conditioning signals are not ``None``;
        # we fail loudly if this is the case.
        if z_source is None and z_cond is None:
            return None
        # Now we handle the returned tensor.
        # Only condition is provided.
        # It is sufficient to check whether ``z_source`` is ``None``,
        # considering the sanity checks performed at the beginning of the function.
        elif z_source is None:
            return z_cond
        # Only source is provided.
        # The reasoning applies symmetrically here: if ``z_cond`` is ``None``, it will
        # necessary follow that ``z_source`` is not None.
        elif z_cond is None:
            return z_source
        # Both source and condition provided.
        # We need to make sure the two data source are concatenable -- the shape
        # contract might be different between the two.
        # NOTE: Think about whether this would break when we have continuous condition
        # data coming from the target distribution.
        return torch.concatenate((make_concatenation_possible(z_cond, z_source, -1), z_source), dim=-1)

    def _encode(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        *,
        cond_dict: dict[str, torch.Tensor] | None = None,
        source: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compose the shared encoder stack → conditioning layer."""
        h_time = self._encode_time(t)
        h_x = self._encode_state(x)
        h_c = self._encode_condition(cond_dict, source)  # returns None if unconditional
        return self._field[_CONDITIONING_LAYER_KEY](h_time, h_x, h_c)


class NeuralVelocity(_NeuralFieldMixin):
    """Time-conditional field: ``v(t, x) -> state_dim``."""

    def forward(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        *,
        condition_dict: dict[str, torch.Tensor] | None = None,
        source: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._field[_FIELD_DECODER_KEY](self._encode(t, x, cond_dict=condition_dict, source=source))

    def get_vf_fn(
        self,
        condition_dict: dict[str, torch.Tensor] | None = None,
        source: torch.Tensor | None = None,
    ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Binds the condition and source, returning ``v(t, x)`` for the ODE/SDE solvers."""
        return partial(self.forward, condition_dict=condition_dict, source=source)


class _BaseNeuralFieldConfig(ModuleConfig):
    """Base configuration set for time-dependent neural fields.

    Fields are time-dependent and implement the ``Field`` protocol:

    ```python
    def forward(
        t: torch.Tensor,
        x: torch.Tensor,
        *,
        condition_dict: dict[str, torch.Tensor] | None = None,
        source: torch.Tensor | None = None,
    )
    ```
    """

    state_encoder: MLPEncoderConfig | None = None
    """State representation fields.

    The states will be passed through an MLP encoder to extract a richer representation.
    If both ``state_encoder`` and ``state_encoder_output_dim`` are provided;
    they need to either be both ``None`` or both initialized for the constructor to work.
    In the initialization of the MLP feature extractor, the output dimensionality is a free variable that can be arbitrarily set;
    therefore, the state encoder needs to be initialized with an ``MLPEncoderConfig``
    and not an ``MLPConfig`` -- the former allows to set upfront the output dimension,
    while the latter assumes the output dimension to be known only at build time,
    because it is constrained by the rest of the architecture.

    :param state_encoder: The ``MLPEncoderConfig`` for the optional ``MLP`` state encoder.
    """

    condition_encoder: SetEncoderConfig | None = None
    """:param condition_encoder: The configurations for the condition encoder."""
    source_encoder: MLPEncoderConfig | None = None
    """:param: The configurations for the source state encoder."""

    time_features: TimeFeaturesConfig | None = None
    time_encoder: MLPEncoderConfig | None = None
    """Time representation fields.

    Time representations are obtained as the modular composition of a sinusoidal time
    feature extractor -- configured with the ``time_features`` field -- and an ``MLP`` featurizer --
    specified with ``time_encoder``. In the initialization of the MLP feature extractor, the output
    dimensionality is a free variable that can be arbitrarily set; therefore, the time encoder
    needs to be initialized with an ``MLPEncoderConfig`` and not an ``MLPConfig`` -- the former
    allows to set upfront the output dimension, while the latter assumes the output
    dimension to be known only at build time, because it is constrained by the rest of the architecture.

    When neither are provided, the model will process raw scalar time indices.

    :param time_features: Optional ``TimeFeaturesConfig`` configuring the used time features.
        When ``None`` no time features will be extracted and the time indices will be processed with
        their raw values.
    :param time_encoder: The ``MLPEncoderConfig`` for the optional ``MLP`` time encoder.
    """

    conditioning: BaseConditioningLayerConfig | None = None
    """:param conditioning: Configurations specifying the conditioning layer to be used.
        Defaults to ``ConcatConditioningConfig``."""

    field_decoder: MLPConfig | None = None
    """:param field_decoder: The ``MLPConfig`` for the field decoder."""


@component("nn.models.velocity", builds=NeuralVelocity)
class NeuralVelocityConfig(_BaseNeuralFieldConfig):
    def build(self, dims: DataDimensions) -> NeuralVelocity:
        return NeuralVelocity(self, dims=dims)
