import abc
from dataclasses import field as dc_field
from typing import Any, Literal

from flax.training import train_state
import flax.linen as nn
import jax.numpy as jnp
import optax

from sc_flow._constants import (
    DEFAULT_CONDITION_ENCODER_OUTPUT_DIM,
    DEFAULT_NUM_TIME_FEATURES,
    DEFAULT_SOURCE_ENCODER_OUTPUT_DIM,
    DEFAULT_TIME_FEATURES_MAX_PERIOD,
    DEFAULT_VF_LATENT_STATE_DIM,
    DEFAULT_VF_LATENT_TIME_DIM,
)
from sc_flow._types import ConditioningLayersId, LayersDict, NestedLayersDict, TimeFeaturesId
from sc_flow.backends.jax._types import ArrayLike, MappedArray, TConditioningFn, TTimeFeaturesFn, TVfFn
from sc_flow.backends.jax._utils import make_concatenation_possible
from sc_flow.backends.jax.nn._conditioning_layers import BaseConditioningLayer, get_conditioning_layer
from sc_flow.backends.jax.nn._modules import BaseModule, FunctionalModule
from sc_flow.backends.jax.nn._set_encoder import SetEncoder
from sc_flow.backends.jax.nn._time_features import get_time_features_fn
from sc_flow.backends.jax.nn._utils import init_module_from_dict

__all__ = [
    "BaseVelocityField",
    "MLPVelocity",
]


class BaseVelocityField(abc.ABC, nn.Module):
    """Base class for neural velocity fields."""

    @abc.abstractmethod
    def _make_vf(
        self,
    ) -> nn.Module:
        """Initializes the neural components of the velocity field."""

    @abc.abstractmethod
    def __call__(
        self,
        t: ArrayLike,
        xt: ArrayLike,
        *args,
        **kwargs,
    ) -> ArrayLike:
        """Performs a forward computation pass on the neural velocity field.

        :param t: The current time index at which the velocity field is computed.
        :type t: class: `~sc_flow.backends.jax._types.ArrayLike`

        :param x: The current state at which the velocity field is computed.
        :type x: class: `~sc_flow.backends.jax._types.ArrayLike`

        :param args: Additional positional arguments used to compute the velocity field.
        :type args: class: `tuple[sc_flow.backends.jax._types.ArrayLike]`
        """

    @abc.abstractmethod
    def get_vf_fn(
        self,
        *args,
        **kwargs,
    ) -> TVfFn:
        """Compiles the velocity field function to be fed to external solvers."""


class MLPVelocity(BaseVelocityField):
    """Class for MLP-base unconditional neural velocity fields.

    The architecture of unconditional velocity fields is defined as follows:
        * (Optional) Time Featurization. An identity mapping is instantiated otherwise.
        * (Optional) Time MLP Encoder. An identity mapping is instantiated otherwise.
        * (Optional) State MLP Encoder. An identity mapping is instantiated otherwise.
        * Conditioning layer, whereby the state and time information are combined.
        * Velocity Field Decoder ultimately computing the velocity field.


    Initializes the velocity field with the given settings.

    :param state_dim: The dimensionality of the state space where the
        dynamics is defined.
    :type state_dim: class: `int`

    :param encode_state: (Optional) Whether to initilize an :class: `MLP` for
        encoding the states. Defaults to `True`.
    :type encode_state: class: `bool`

    :param encode_time: (Optional) Whether to initilize an :class: `MLP` for
        encoding the time index. Defaults to `True`.
    :type encode_time: class: `bool`

    :param time_features_id: (Optional) The identifier for the chosen time features.
        Could be set to any of the string identifiers for predefined time features,
        specified in :class: `TimeFeaturesId`. Ignored when :param: `time_features_fn`
        is passed as well. Defaults to `None`, in which case no time features are computed
        (i.e.: identity mapping).
    :type time_features_id: class: `TimeFeaturesId | None`

    :param time_features_fn: (Optional) Function defining the custom time featurizer which should accept tensor with
        trailing singleton dimension and expand the trailing dimension to $2K$.The input function is wrapped,
        so that it only need to actually implement the expansion of the trailing dimension.
        When provided, takes precedence over string initialization. Defaults to `None`.
    :type time_features_fn: class: `TTimeFeaturesFn | None`

    :param num_time_features: (Optional) Sets the number of resulting time features, hence it must be even.
        Raises a :class: `ValueError` otherwise. When not provided, it will be set to
        :constant: `sc_flow._constants.DEFAULT_NUM_TIME_FEATURES`. Defaults to `None`.
    :type num_time_features: class: `int | None`

    :param max_period: (Optional) Sets the value of $M$, used for the linear scaling of the time features.
        Only used when :param: `time_features_id` is set to `"torch-cfm"`, ignored otherwise.
        When not provided, it will be set to :constant: `sc_flow._constants.DEFAULT_TIME_FEATURES_MAX_PERIOD`. Defaults to `None`.
    :type max_period: class: `int | None`

    :param time_features_kwargs: (Optional) Keyword arguments used to compile custom time featurizers. Only used when
        initializing time featurizers with :param: `time_features_fn`, ignored otherwise. Defaults to `None`
    :type time_features_kwargs: class: `dict[str, Any] | None`

    :param state_encoder_output_dim: (Optional) The output dimensionality for the state encoder, only used when :param: `encode_state`
        is set to `True`. When `None` it will be set to :constant: `sc_flow._constants.DEFAULT_VF_LATENT_STATE_DIM`.
        Defaults to `None`.
    :type state_encoder_output_dim: class: `int | None`

    :param time_encoder_output_dim: (Optional) The output dimensionality for the time encoder, only used when :param: `encode_time`
        is set to `True`. When `None` it will be set to :constant: `sc_flow._constants.DEFAULT_VF_LATENT_TIME_DIM`.
        Defaults to `None`.
    :type time_encoder_output_dim: class: `int | None`

    :param state_encoder_mlp_kwargs: (Optional) Keyword arguments used for initializing the state encoder. When used, its key-value pairs
        should match the signature of the `__init__` method of the :class: `MLP` class, raise :class: `TypeError` otherwise.
        Only used when :param: `encode_state` is set to `True`. When `None` it will be set to an empty dictionary,
        hence falling back to the default configurations defined directly in :class: `MLP`. Defaults to `None`.
    :type state_encoder_mlp_kwargs: class: `dict[str, Any] | None`

    :param time_encoder_mlp_kwargs: (Optional) Keyword arguments used for initializing the time encoder. When used, its key-value pairs
        should match the signature of the `__init__` method of the :class: `MLP` class, raise :class: `TypeError` otherwise.
        Only used when :param: `encode_time` is set to `True`. When `None` it will be set to an empty dictionary,
        hence falling back to the default configurations defined directly in :class: `MLP`. Defaults to `None`.
    :type time_encoder_mlp_kwargs: class: `dict[str, Any] | None`

    :param vf_decoder_mlp_kwargs: (Optional) Keyword arguments used for initializing the velocity field decoder. Its key-value pairs
        should match the signature of the `__init__` method of the :class: `MLP` class, raise :class: `TypeError` otherwise.
        When `None` it will be set to an empty dictionary, hence falling back to the default configurations
        defined directly in :class: `MLP`. Defaults to `None`.
    :type vf_decoder_mlp_kwargs: class: `dict[str, Any] | None`

    :param conditioning_id: (Optional) String identifier indicating the type of conditioning applied to the states.
        For unconditional velocity fields, the conditioning is only done with respect to the time index.
        Could be set to any of the string identifiers for predefined time features, specified in :class: `ConditioningLayersId`.
        Ignored when :param: `conditioning_fn` is passed as well. Defaults to `None` and, when not provided, it will be set internally set to
        :constant: `sc_flow._constants.DEFAULT_CONDITIONING_LAYER`.
    :type conditioning_id: class: `ConditioningLayersId`

    :param conditioning_fn: (Optional) Callable using for the instantiation of custom conditioning layers.
        When provided, takes precedence over string initialization. Defaults to `None`.
    :type conditioning_fn: class: `TConditioningFn`

    :param conditioning_kwargs: (Optional) Keyword arguments used to initialize the conditioning layer.
        Ignored when the using concatenation based conditioning. When setting :param: `conditioning_id`
        to `"resnet1d"`, it should match the signatures of the `__init__` method of the :class: `Resnet1d` class,
        raise :class: `TypeError` otherwise. Defaults to `None`.
    :type conditioning_kwargs: class: `dict[str, Any]`

    :param condition_encoder_input_layers: Dictionary mapping each perturbation covariate
        identifier to the configurations for their respective input layer.
    :type condition_encoder_input_layers: class: `NestedLayersDict`

    :param condition_encoder_output_dim: The output dimensionality of the set encoder.
    :type condition_encoder_output_dim: class: `int`

    :param condition_encoder_pooling_mode: Identifier for the pooling strategy of conditioning covariates.
        Defaults to `"mean"`.
    :type condition_encoder_pooling_mode: class: `Literal["mean", "sum"]`

    :param condition_encoder_pooling_kwargs: Optional keyword arguments for pooling layer.
        Ignored when pooling is `"mean"` or `"sum"`, defaults to `None`.
    :type condition_encoder_pooling_kwargs: class: `dict[str, Any]`

    :param condition_encoder_covariates_not_pooled: Optional sequence of the perturbation covariate
        identifiers for which the pooling should not be applied. Defaults to `None`.
    :class condition_encoder_covariates_not_pooled: class: `Sequence[int] | None`

    :param condition_encoder_output_layers_kwargs: Dictionary containing the configurations for the output layer.
        Defaults to `None`.
    :type condition_encoder_output_layers_kwargs: class: `LayersDict | None`

    :param source_encoder_mlp_kwargs: (Optional) Keyword arguments used to initialize
        source state encoders when provided. This is needed to retain the information about the
        source states when interpolating between noise and samples from the target distribution.
        Defaults to `None`, in which case no source encoder will be initialized.
    :type source_encoder_mlp_kwargs: class: `LayersDict`

    :param source_encoder_output_dim: (Optional) Output dimensionality for the source state encoder.
        Only used when encoding source states. Defaults to `None`.
    :type source_encoder_output_dim: class: `int`
    """

    state_dim: int
    encode_state: bool = True
    encode_time: bool = True
    time_features_id: TimeFeaturesId | None = None
    time_features_fn: TTimeFeaturesFn | None = None
    num_time_features: int | None = DEFAULT_NUM_TIME_FEATURES
    max_period: int = DEFAULT_TIME_FEATURES_MAX_PERIOD
    time_features_kwargs: dict[str, Any] = dc_field(default_factory=dict)
    state_encoder_output_dim: int = DEFAULT_VF_LATENT_STATE_DIM
    time_encoder_output_dim: int = DEFAULT_VF_LATENT_TIME_DIM
    state_encoder_mlp_kwargs: LayersDict = dc_field(default_factory=dict)
    time_encoder_mlp_kwargs: LayersDict = dc_field(default_factory=dict)
    vf_decoder_mlp_kwargs: LayersDict = dc_field(default_factory=dict)
    conditioning_id: ConditioningLayersId | None = "concat"
    conditioning_fn: TConditioningFn | None = None
    conditioning_kwargs: dict[str, Any] | None = None
    condition_encoder_input_layers: NestedLayersDict | None = None
    condition_encoder_output_dim: int = DEFAULT_CONDITION_ENCODER_OUTPUT_DIM
    condition_encoder_pooling_mode: Literal["mean", "sum"] = "mean"
    condition_encoder_pooling_kwargs: dict[str, Any] = dc_field(default_factory=dict)
    condition_encoder_output_layers_kwargs: LayersDict | None = None
    source_encoder_mlp_kwargs: LayersDict | None = None
    source_encoder_output_dim: int = DEFAULT_SOURCE_ENCODER_OUTPUT_DIM

    def setup(self) -> None:
        """Initialize the network."""
        self.vf = self._make_vf()

    @property
    def use_time_features(self) -> bool:
        """Boolean flag indicating whether time featurization is used for the current velocity field.

        This is the case when either :param: `time_features_id` or :param: `time_features_fn`
        are passed during initialization.
        """
        return (self.time_features_id is not None) or (self.time_features_fn is not None)

    @property
    def use_source_encoder(self) -> bool:
        """Boolean flag indicating whether an MLP should be initialized to encode source states."""
        return self.source_encoder_mlp_kwargs is not None

    @property
    def source_encoder_input_dim(self) -> int:
        """Returns the input dimensionality for the source encoder.

        When not explicitly specified in source_encoder_mlp_kwargs, it will be
        automatically retrieved from state_dim.
        """
        input_dim = self.source_encoder_mlp_kwargs.get("input_dim", None)
        if input_dim is None:
            return self.state_dim
        return input_dim

    @property
    def conditioning_dim(self) -> int | None:
        """Returns the dimensionality for the condition input of the conditioning layers."""
        if self.is_conditional and self.use_source_encoder:
            return self.resolved_condition_encoder_output_dim + self.resolved_source_encoder_output_dim
        elif self.is_conditional and (not self.use_source_encoder):
            return self.resolved_condition_encoder_output_dim
        elif self.use_source_encoder:
            return self.resolved_source_encoder_output_dim
        return None

    def get_num_time_features(
        self,
    ) -> int:
        """Returns the number of time features.

        If no time featurization is used, it will return simply 1 (i.e.: scalar time)
        Otherwise, when the number of time feature is not provided, it will fall back to the default defined in
        :constant: `sc_flow._constants.DEFAULT_NUM_TIME_FEATURES`.
        """
        if not self.use_time_features:
            return 1
        else:
            if self.num_time_features is None:
                return DEFAULT_NUM_TIME_FEATURES
            return self.num_time_features

    def get_encoded_conditions(
        self,
        condition_dict: MappedArray | None,
    ) -> ArrayLike | None:
        """Retrieves the encoded conditions."""
        if self.is_conditional:
            if condition_dict is None:
                msg = "Conditional VFs should take a condition as input, found `None`."
                raise TypeError(msg)
            return self.vf["condition_encoder"](condition_dict)
        return None

    def get_encoded_source(
        self,
        source: ArrayLike | None,
    ) -> ArrayLike | None:
        """Retrieves the encoded source states."""
        if self.use_source_encoder:
            if source is None:
                msg = "When using source encoder a source state should be passed, found `None`."
                raise TypeError(msg)
            return self.vf["source_encoder"](source)
        return None

    def get_conditioning_input(self, encoded_condition: ArrayLike, encoded_source: ArrayLike) -> ArrayLike | None:
        """Retrieves the condition input for the conditioning layers."""
        if encoded_condition is not None and encoded_source is not None:
            return jnp.concatenate(
                (make_concatenation_possible(encoded_condition, encoded_source, -1), encoded_source), dim=-1
            )
        elif encoded_condition is None and encoded_source is not None:
            return encoded_source
        elif encoded_condition is not None and encoded_source is None:
            return encoded_source
        return None

    def _make_time_features(
        self,
    ) -> FunctionalModule:
        """Initializes the time features.

        Wraps a :class: `FunctionalModule` around the function used to compute the time features for
        compatibility with Flax's module system.
        """
        time_features_fn = get_time_features_fn(
            num_time_features=self.get_num_time_features(),
            time_features_id=self.time_features_id,
            time_features_fn=self.time_features_fn,
            max_period=self.max_period,
            time_features_kwargs=self.time_features_kwargs,
        )
        return FunctionalModule(time_features_fn)

    def _make_time_encoder(
        self,
    ) -> BaseModule:
        """Initializes the optional time encoder.

        When :param: `encode_time` is set to `True` it will initialize a :class: `MLP` with the configurations
        specified in :param: `time_encoder_mlp_kwargs`. Returns an identity function wrapper otherwise otherwise.
        """
        if self.encode_time:
            return init_module_from_dict(
                self.time_encoder_mlp_kwargs,
                input_dim=self.get_num_time_features(),
                output_dim=self.time_encoder_output_dim,
            )
        return FunctionalModule(lambda x: x)

    def _make_state_encoder(
        self,
    ) -> BaseModule:
        """Initializes the optional state encoder.

        When :param: `encode_state` is set to `True` it will initialize a :class: `MLP` with the configurations
        specified in :param: `state_encoder_mlp_kwargs`. Returns an identity function wrapper otherwise otherwise.
        """
        if self.encode_state:
            return init_module_from_dict(
                self.state_encoder_mlp_kwargs,
                input_dim=self.state_dim,
                output_dim=self.state_encoder_output_dim,
            )
        return FunctionalModule(lambda x: x)

    def _make_conditioning_layer(
        self,
    ) -> BaseConditioningLayer:
        """Initializes the conditioning layer according to the configurations specified during initialization."""
        return get_conditioning_layer(
            self.state_encoder_output_dim if self.encode_state else self.state_dim,
            self.time_encoder_output_dim if self.encode_time else self.get_num_time_features(),
            latent_condition_dim=self.conditioning_dim,
            conditioning_id=self.conditioning_id,
            conditioning_fn=self.conditioning_fn,
            conditioning_kwargs=self.conditioning_kwargs,
        )

    def _make_condition_encoder(
        self,
    ) -> BaseModule:
        """Initializes the condition encoder when the required settings are specified."""
        if not self.is_conditional:
            msg = (
                "To initialize the condition encoder you have to pass"
                "the `condition_encoder_input_layers` argument, `None` found."
            )
            raise TypeError(msg)
        return SetEncoder(
            self.condition_encoder_input_layers,
            self.condition_encoder_output_dim,
            pooling_mode=self.condition_encoder_pooling_mode,
            pooling_kwargs=self.condition_encoder_pooling_kwargs,
            output_layers_kwargs=self.condition_encoder_output_layers_kwargs,
        )

    def _make_vf_decoder(
        self,
        decoder_input_dim: int,
    ) -> BaseModule:
        """Initializes the velocity field decoder.

        It will initialize a :class: `MLP` with the configurations specified in :param: `vf_decoder_mlp_kwargs`.
        """
        return init_module_from_dict(
            self.vf_decoder_mlp_kwargs,
            input_dim=decoder_input_dim,
            output_dim=self.state_dim,
        )

    def _make_source_encoder(
        self,
    ) -> BaseModule:
        """Initializes the source state encoder when the required settings are passed."""
        if not self.use_source_encoder:
            msg = (
                "To initialize the source encoder you have to pass"
                "the `source_encoder_mlp_kwargs` argument, `None` found."
            )
            raise TypeError(msg)

        return init_module_from_dict(
            self.source_encoder_mlp_kwargs,
            input_dim=self.source_encoder_input_dim,
            output_dim=self.source_encoder_output_dim,
        )

    def _make_vf(
        self,
    ) -> nn.Module:
        """Initializes the neural components of the velocity field.

        This is done by calling the `self._make_*` methods defined above.
        """
        modules = {
            "time_features": self._make_time_features(),
            "time_encoder": self._make_time_encoder(),
            "state_encoder": self._make_state_encoder(),
            "conditioning_layer": self._make_conditioning_layer(),
        }
        if self.is_conditional:
            modules["condition_encoder"] = self._make_condition_encoder()
        if self.use_source_encoder:
            modules["source_encoder"] = self._make_source_encoder()
        modules["vf_decoder"] = self._make_vf_decoder(
            modules["conditioning_layer"].output_dim,
        )
        return modules

    def __call__(
        self,
        t: ArrayLike,
        x: ArrayLike,
        condition_dict: MappedArray | None = None,
        source: ArrayLike | None = None,
    ) -> ArrayLike:
        """Performs a forward computation pass on the neural velocity field.

        :param t: The current time index at which the velocity field is computed.
        :type t: class: `ArrayLike`

        :param x: The current state at which the velocity field is computed.
        :type x: class: `ArrayLike`

        :param condition_dict: The input dictionary containing the data for
            each perturbation covariate.
        :type condition_dict: class: `MappedArray`
        """
        encoded_xt = self.vf["state_encoder"](x)

        encoded_t = self.vf["time_features"](t)
        encoded_t = self.vf["time_encoder"](encoded_t)

        encoded_condition = self.get_encoded_conditions(condition_dict)
        encoded_source = self.get_encoded_source(source)

        encoded_concat = self.vf["conditioning_layer"](
            encoded_t, encoded_xt, encoded_condition=self.get_conditioning_input(encoded_condition, encoded_source)
        )
        return self.vf["vf_decoder"](encoded_concat)

    def get_vf_fn(
        self,
        condition_dict: MappedArray | None = None,
        source: ArrayLike | None = None,
    ) -> TVfFn:
        """Compiles the velocity field function to be fed to external solvers."""

        def _vf_fn(t: ArrayLike, x: ArrayLike):
            return self(t, x, condition_dict=condition_dict, source=source)

        return _vf_fn

    def create_train_state(
        self,
        rng: jax.Array,
        optimizer: optax.OptState,
        input_dim: int,
        condition_dict: dict[str, jnp.ndarray],
    ) -> train_state.TrainState:
        """Create the training state for the velocity field.

        :param rng: A random number generator.
        :type rng: class: `jax.Array`

        :param optimizer: The optimizer.
        :type optimizer: class: `optax.OptState`

        :param input_dim: The input dimensionality of the velocity field.
        :type args: class: `int`

        :param conditions: The condition dictionary.
        :type conditions: class: `dict[str, jnp.ndarray]`
        """
        t, x_t = jnp.ones((1, 1)), jnp.ones((1, input_dim))
        # TODO use max_combination_length?
        # cond = {
        #     pert_cov: jnp.ones((1, self.max_combination_length, condition.shape[-1]))
        #     for pert_cov, condition in conditions.items()
        # }
        condition_dict = condition_dict
        params = self.init(
            rng,
            t,
            x_t,
            condition_dict=condition_dict
        )["params"]
        return train_state.TrainState.create(apply_fn=self.apply, params=params, tx=optimizer)

    @property
    def is_conditional(
        self,
    ) -> bool:
        """Whether a condition encoder is associated to velocity field."""
        return self.condition_encoder_input_layers is not None
