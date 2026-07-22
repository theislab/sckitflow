import abc
from dataclasses import dataclass, field
from typing import Any

import torch

from sc_flow._constants import (
    DEFAULT_NUM_TIME_FEATURES,
    DEFAULT_SOURCE_ENCODER_OUTPUT_DIM,
    DEFAULT_TIME_FEATURES_MAX_PERIOD,
    DEFAULT_VF_LATENT_STATE_DIM,
    DEFAULT_VF_LATENT_TIME_DIM,
)
from sc_flow._types import CombinerId, LayersDict, TimeFeaturesId
from sc_flow.core._torch_types import MappedTensor, VelocityFieldFn
from sc_flow.core._torch_utils import make_concatenation_possible
from sc_flow.core.nn._modules import BaseModule, FunctionalModule
from sc_flow.core.nn._utils import init_module_from_dict
from sc_flow.flow._combiner import BaseCombiner, get_combiner
from sc_flow.flow._set_encoder import SetEncoder
from sc_flow.flow._time_features import get_time_features_fn

__all__ = [
    "BaseVelocityField",
    "MLPEmbedderConfig",
    "MLPVelocity",
]


@dataclass
class MLPEmbedderConfig:
    """Config for the optional MLP that embeds one vector stream into a latent width.

    Used for the velocity field's ``state_embedder`` / ``time_embedder`` / ``source_embedder`` slots.
    ``None`` in a slot skips the embedder and passes that stream through raw. The velocity field supplies
    the MLP ``input_dim`` (it knows each stream's width), so this config only carries the target
    ``output_dim`` (``None`` → a per-stream default) and any extra :class:`~sc_flow.core.nn.MLP` kwargs.
    """

    output_dim: int | None = None
    mlp_kwargs: dict[str, Any] = field(default_factory=dict)


class BaseVelocityField(BaseModule):
    """Base class for neural velocity fields."""

    @abc.abstractmethod
    def forward(
        self,
        t: torch.Tensor,
        xt: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Performs a forward computation pass on the neural velocity field.

        :param t: The current time index at which the velocity field is computed.
        :type t: class: `torch.Tensor`

        :param x: The current state at which the velocity field is computed.
        :type x: class: `torch.Tensor`

        :param args: Additional positional arguments used to compute the velocity field.
        :type args: class: `tuple[torch.Tensor]`
        """

    @abc.abstractmethod
    def get_vf_fn(
        self,
        *args,
        **kwargs,
    ) -> VelocityFieldFn:
        """Compiles the velocity field function to be fed to external solvers."""


class MLPVelocity(BaseVelocityField):
    """Class for MLP-base unconditional neural velocity fields.

    Torch port of cellflow's ``ConditionalVelocityField`` / ``GENOTConditionalVelocityField`` (theislab/cellflow,
    ``src/cellflow/networks/_velocity_field.py``, flax): same time / state / condition / source encoders,
    combiner (concatenation / FiLM / resnet) and decoder; :meth:`condition_stats` mirrors cellflow's
    ``get_condition_embedding``. Kept structurally aligned so the jax original and this torch port stay mutually
    reviewable.

    The architecture of unconditional velocity fields is defined as follows:
        * (Optional) Time Featurization. An identity mapping is instantiated otherwise.
        * (Optional) Time MLP Encoder. An identity mapping is instantiated otherwise.
        * (Optional) State MLP Encoder. An identity mapping is instantiated otherwise.
        * Combiner, whereby the state and time information are combined.
        * Velocity Field Decoder ultimately computing the velocity field.
    """

    def __init__(
        self,
        state_dim: int,
        state_embedder: MLPEmbedderConfig | None = None,
        time_embedder: MLPEmbedderConfig | None = None,
        source_embedder: MLPEmbedderConfig | None = None,
        time_features_id: TimeFeaturesId | None = None,
        num_time_features: int | None = None,
        max_period: int | None = None,
        vf_decoder_mlp_kwargs: LayersDict | None = None,
        combiner: CombinerId | BaseCombiner | None = None,
        condition_encoder: SetEncoder | None = None,
    ) -> None:
        """Initializes the velocity field with the given settings.

        :param state_dim: The dimensionality of the state space where the
            dynamics is defined.
        :type state_dim: class: `int`

        :param state_embedder: (Optional) :class: `MLPEmbedderConfig` for the MLP that embeds the state
            ``x`` into a latent width (default output dim :constant: `DEFAULT_VF_LATENT_STATE_DIM`).
            ``None`` passes the raw state through.
        :type state_embedder: class: `MLPEmbedderConfig | None`

        :param time_embedder: (Optional) :class: `MLPEmbedderConfig` for the MLP that embeds the time
            features into a latent width (default output dim :constant: `DEFAULT_VF_LATENT_TIME_DIM`).
            ``None`` passes the raw time features through.
        :type time_embedder: class: `MLPEmbedderConfig | None`

        :param source_embedder: (Optional) :class: `MLPEmbedderConfig` for the MLP that embeds the source
            state (used by GENOT to condition on the source cell; default output dim
            :constant: `DEFAULT_SOURCE_ENCODER_OUTPUT_DIM`). ``None`` (default) means no source embedder.
        :type source_embedder: class: `MLPEmbedderConfig | None`

        :param time_features_id: (Optional) The identifier for the chosen time features.
            Could be set to any of the string identifiers for predefined time features,
            specified in :class: `TimeFeaturesId` (``"sinusoidal"`` or ``"log-sinusoidal"``).
            Defaults to `None`, in which case no time features are computed (i.e.: identity mapping).
        :type time_features_id: class: `TimeFeaturesId | None`

        :param num_time_features: (Optional) Sets the number of resulting time features, hence it must be even.
            Raises a :class: `ValueError` otherwise. When not provided, it will be set to
            :constant: `sc_flow._constants.DEFAULT_NUM_TIME_FEATURES`. Defaults to `None`.
        :type num_time_features: class: `int | None`

        :param max_period: (Optional) Sets the value of $M$, used for the log scaling of the time features.
            Only used when :param: `time_features_id` is set to `"log-sinusoidal"`, ignored otherwise.
            When not provided, it will be set to :constant: `sc_flow._constants.DEFAULT_TIME_FEATURES_MAX_PERIOD`. Defaults to `None`.
        :type max_period: class: `int | None`

        :param vf_decoder_mlp_kwargs: (Optional) Keyword arguments used for initializing the velocity field decoder. Its key-value pairs
            should match the signature of the `__init__` method of the :class: `MLP` class, raise :class: `TypeError` otherwise.
            When `None` it will be set to an empty dictionary, hence falling back to the default configurations
            defined directly in :class: `MLP`. Defaults to `None`.
        :type vf_decoder_mlp_kwargs: class: `dict[str, Any] | None`

        :param combiner: (Optional) How state/time (and any condition) are combined. Either a built-in
            choice by name from :class: `CombinerId` (``"concat"`` or ``"resnet1d"``), or your own
            :class: `BaseCombiner` subclass instance sized for this VF's latent dims. A built-in
            name is saved to and restored from ``config.json``; a custom instance is not saved and must be
            passed again to ``from_pretrained``. Defaults to `None`, falling back to
            :constant: `sc_flow._constants.DEFAULT_COMBINER`.
        :type combiner: class: `CombinerId | BaseCombiner | None`

        :param condition_encoder: (Optional) The perturbation-covariate condition encoder — a
            :class: `SetEncoder` instance (built directly, since its dims come from the data, not from this
            velocity field). ``None`` makes the field unconditional. This runtime object is not yet a nested
            component spec, so portable ``save_pretrained`` export is disabled for a conditional velocity
            field until the enclosing architecture config can reconstruct it. Trusted training checkpoints
            remain available.
        :type condition_encoder: class: `SetEncoder | None`
        """
        super().__init__()
        self._state_dim = state_dim
        # Each stream embedder is presence-based: an MLPEmbedderConfig builds an MLP over that stream, None
        # passes the raw stream through. state/time default off here; the FlowMatching facade turns them on.
        self._state_embedder = state_embedder
        self._time_embedder = time_embedder
        self._source_embedder = source_embedder
        self._time_features_id = time_features_id
        self._num_time_features = num_time_features
        self._max_period = DEFAULT_TIME_FEATURES_MAX_PERIOD if max_period is None else max_period
        self._vf_decoder_mlp_kwargs = {} if vf_decoder_mlp_kwargs is None else vf_decoder_mlp_kwargs
        # A built-in name (str) is saved/restored via config.json; a custom BaseCombiner instance
        # is not saved (see BaseModule) and must be passed again to from_pretrained — otherwise the strict
        # weight load raises a clear mismatch error.
        self._combiner = combiner
        self._condition_encoder = condition_encoder

        self._vf = self._make_modules()

    @property
    def _use_time_features(
        self,
    ) -> bool:
        """Boolean flag indicating whether time featurization is used for the current velocity field.

        This is the case when :param: `time_features_id` is passed during initialization.
        """
        return self._time_features_id is not None

    @property
    def use_source_embedder(
        self,
    ) -> bool:
        """Whether a source-state embedder is configured."""
        return self._source_embedder is not None

    @staticmethod
    def _embed_output_dim(config: MLPEmbedderConfig | None, raw_dim: int, default_output_dim: int) -> int:
        """Width of a stream after its optional embedder: the raw width when ``None``, else the config output."""
        if config is None:
            return raw_dim
        return default_output_dim if config.output_dim is None else config.output_dim

    @property
    def _source_embedder_input_dim(self) -> int:
        """Input width for the source embedder (``mlp_kwargs['input_dim']`` if given, else ``state_dim``)."""
        input_dim = self._source_embedder.mlp_kwargs.get("input_dim") if self._source_embedder is not None else None
        return self._state_dim if input_dim is None else input_dim

    @property
    def _source_embedder_output_dim(self) -> int:
        """Output width of the source embedder."""
        return self._embed_output_dim(
            self._source_embedder, self._source_embedder_input_dim, DEFAULT_SOURCE_ENCODER_OUTPUT_DIM
        )

    @property
    def _combiner_dim(
        self,
    ) -> int | None:
        """Latent width of the condition input fed to the combiner (condition embedding, plus source, or None)."""
        cond = self._condition_encoder.output_dim if self.is_conditional else None
        src = self._source_embedder_output_dim if self.use_source_embedder else None
        if cond is not None and src is not None:
            return cond + src
        return cond if cond is not None else src

    def _get_num_time_features(
        self,
    ) -> int:
        """Returns the number of time features.

        If no time featurization is used, it will return simply 1 (i.e.: scalar time)
        Otherwise, when the number of time feature is not provided, it will fall back to the default defined in
        :constant: `sc_flow._constants.DEFAULT_NUM_TIME_FEATURES`.
        """
        if not self._use_time_features:
            return 1
        else:
            if self._num_time_features is None:
                return DEFAULT_NUM_TIME_FEATURES
            return self._num_time_features

    def condition_stats(
        self,
        condition_dict: MappedTensor | None = None,
        condition_mask: MappedTensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """The condition encoder outputs ``(mean, logvar)`` (``logvar`` is ``None`` when deterministic).

        Returns ``(None, None)`` for an unconditional field. A training objective calls this **once** and
        reuses the result for both the velocity (:meth:`velocity_from_embedding`) and the encoder
        regularization — which is also what makes a stochastic encoder's reparameterization correct (a
        single noise draw shared by the velocity and the KL term).
        """
        if not self.is_conditional:
            return None, None
        if condition_dict is None:
            msg = "Conditional VFs should take a condition as input, found `None`."
            raise TypeError(msg)
        return self._vf["condition_encoder"](condition_dict, condition_mask=condition_mask)

    def _get_encoded_source(
        self,
        source: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Retrieves the encoded source states."""
        if self.use_source_embedder:
            if source is None:
                msg = "When using a source embedder a source state should be passed, found `None`."
                raise TypeError(msg)
            return self._vf["source_embedder"](source)
        return None

    def _get_combiner_input(
        self, encoded_condition: torch.Tensor, encoded_source: torch.tensor
    ) -> torch.Tensor | None:
        """Retrieves the condition input for the combiner layers."""
        if encoded_condition is not None and encoded_source is not None:
            return torch.concatenate(
                (make_concatenation_possible(encoded_condition, encoded_source, -1), encoded_source), dim=-1
            )
        elif encoded_condition is None and encoded_source is not None:
            return encoded_source
        elif encoded_condition is not None and encoded_source is None:
            return encoded_condition
        return None

    def _make_time_features(
        self,
    ) -> FunctionalModule:
        """Initializes the time features.

        Wraps a :class: `FunctionalModule` around the function used to compute the time features for
        compatibility with `pytorch`'s `torch.nn.ModuleDict`.
        """
        time_features_fn = get_time_features_fn(
            num_time_features=self._get_num_time_features(),
            time_features_id=self._time_features_id,
            max_period=self._max_period,
        )
        return FunctionalModule(time_features_fn)

    def _make_embedder(
        self,
        config: MLPEmbedderConfig | None,
        input_dim: int,
        default_output_dim: int,
    ) -> BaseModule | torch.nn.Identity:
        """Builds a stream embedder MLP from its config, or :class:`torch.nn.Identity` when the slot is ``None``."""
        if config is None:
            return torch.nn.Identity()
        output_dim = default_output_dim if config.output_dim is None else config.output_dim
        return init_module_from_dict(dict(config.mlp_kwargs), input_dim=input_dim, output_dim=output_dim)

    @property
    def _combiner_dims(self) -> tuple[int, int, int | None]:
        """The ``(latent_state, latent_time, latent_condition)`` dims the combiner layer must accept."""
        return (
            self._embed_output_dim(self._state_embedder, self._state_dim, DEFAULT_VF_LATENT_STATE_DIM),
            self._embed_output_dim(self._time_embedder, self._get_num_time_features(), DEFAULT_VF_LATENT_TIME_DIM),
            self._combiner_dim,
        )

    def _make_combiner(
        self,
    ) -> BaseCombiner:
        """Initializes the combiner layer.

        Uses a custom :class:`BaseCombiner` instance directly (validating it was sized for this
        VF's latent dims), otherwise builds a built-in layer from the name (defaulting when ``None``).
        """
        latent_state_dim, latent_time_dim, latent_condition_dim = self._combiner_dims
        if isinstance(self._combiner, BaseCombiner):
            got = (
                self._combiner._latent_state_dim,
                self._combiner._latent_time_dim,
                self._combiner._latent_condition_dim,
            )
            if got != (latent_state_dim, latent_time_dim, latent_condition_dim):
                msg = (
                    f"Custom combiner was sized for (state, time, condition)={got}, but this velocity "
                    f"field feeds it {(latent_state_dim, latent_time_dim, latent_condition_dim)}."
                )
                raise ValueError(msg)
            return self._combiner
        return get_combiner(
            latent_state_dim,
            latent_time_dim,
            latent_condition_dim=latent_condition_dim,
            combiner_id=self._combiner,
        )

    def _make_vf_decoder(
        self,
        decoder_input_dim: int,
    ) -> BaseModule:
        """Initializes the velocity field decoder.

        It will initialize a :class: `MLP` with the configurations specified in :param: `vf_decoder_mlp_kwargs`.
        """
        return init_module_from_dict(
            self._vf_decoder_mlp_kwargs,
            input_dim=decoder_input_dim,
            output_dim=self._state_dim,
        )

    def _make_modules(
        self,
    ) -> torch.nn.Module:
        """Initializes the neural components of the velocity field.

        This is done by calling the `self._make_*` methods defined above.
        """
        modules = {
            "time_features": self._make_time_features(),
            "time_embedder": self._make_embedder(
                self._time_embedder, self._get_num_time_features(), DEFAULT_VF_LATENT_TIME_DIM
            ),
            "state_embedder": self._make_embedder(self._state_embedder, self._state_dim, DEFAULT_VF_LATENT_STATE_DIM),
            "combiner": self._make_combiner(),
        }
        if self.is_conditional:
            modules["condition_encoder"] = self._condition_encoder
        if self.use_source_embedder:
            modules["source_embedder"] = self._make_embedder(
                self._source_embedder, self._source_embedder_input_dim, DEFAULT_SOURCE_ENCODER_OUTPUT_DIM
            )
        modules["vf_decoder"] = self._make_vf_decoder(
            modules["combiner"].output_dim,
        )
        return torch.nn.ModuleDict(modules)

    def forward(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        condition_dict: MappedTensor | None = None,
        source: torch.Tensor | None = None,
        condition_mask: MappedTensor | None = None,
    ) -> torch.Tensor:
        """Performs a forward computation pass on the neural velocity field.

        :param t: The current time index at which the velocity field is computed.
        :type t: class: `torch.Tensor`

        :param x: The current state at which the velocity field is computed.
        :type x: class: `torch.Tensor`

        :param condition_dict: The input dictionary containing the data for
            each perturbation covariate.
        :type condition_dict: class: `MappedTensor`

        :param condition_mask: Optional boolean valid-token mask per condition realm.
        """
        # Inference/solver path: use the condition **mean** embedding (a stochastic encoder's inference
        # is its mean, i.e. encoder_noise = 0). Training reparameterizes upstream and calls
        # :meth:`velocity_from_embedding` directly.
        mean, _ = self.condition_stats(condition_dict, condition_mask=condition_mask)
        return self.velocity_from_embedding(t, x, mean, source=source)

    def velocity_from_embedding(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        cond_embedding: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Velocity from a **precomputed** condition embedding (skips condition encoding).

        Lets a training objective encode the condition once (via :meth:`condition_stats`), reparameterize
        if stochastic, and reuse the embedding here — avoiding a second encoder pass and keeping a
        stochastic encoder's noise draw consistent between the velocity and its KL regularization.
        :meth:`forward` is this composed with :meth:`condition_stats`.
        """
        encoded_xt = self._vf["state_embedder"](x)
        encoded_t = self._vf["time_embedder"](self._vf["time_features"](t))
        encoded_source = self._get_encoded_source(source)
        encoded_concat = self._vf["combiner"](
            encoded_t, encoded_xt, encoded_condition=self._get_combiner_input(cond_embedding, encoded_source)
        )
        return self._vf["vf_decoder"](encoded_concat)

    def get_vf_fn(
        self,
        condition_dict: MappedTensor | None = None,
        source: torch.Tensor | None = None,
        condition_mask: MappedTensor | None = None,
    ) -> VelocityFieldFn:
        """Compiles the velocity field function to be fed to external solvers."""

        def _vf_fn(t: torch.Tensor, x: torch.Tensor):
            return self.forward(
                t,
                x,
                condition_dict=condition_dict,
                source=source,
                condition_mask=condition_mask,
            )

        return _vf_fn

    @property
    def is_conditional(
        self,
    ) -> bool:
        """Whether a condition encoder is associated to velocity field."""
        return self._condition_encoder is not None

    @property
    def is_stochastic(
        self,
    ) -> bool:
        """Whether the condition encoder is variational (a stochastic :class:`SetEncoder`)."""
        return self.is_conditional and self._condition_encoder.is_stochastic
