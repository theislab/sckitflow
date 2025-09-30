import abc
from typing import Any

import torch

from sc_flow._constants import (
    DEFAULT_NUM_TIME_FEATURES,
    DEFAULT_TIME_FEATURES_MAX_PERIOD,
    DEFAULT_VF_LATENT_STATE_DIM,
    DEFAULT_VF_LATENT_TIME_DIM,
)
from sc_flow._types import ConditioningLayersId, TimeFeaturesId
from sc_flow._utils import verify_fn_kwargs_dictionary
from sc_flow.backends.torch._types import TConditioningFn, TTimeFeaturesFn, VfFunction
from sc_flow.backends.torch.nn._conditioning_layers import BaseConditioningLayer, get_conditioning_layer
from sc_flow.backends.torch.nn._modules import MLP, BaseModule, FunctionalModule
from sc_flow.backends.torch.nn._time_features import get_time_features_fn

__all__ = [
    "BaseVelocityField",
    "MLPUnconditionalVF",
]


class BaseVelocityField(abc.ABC, torch.nn.Module):
    """Base class for neural velocity fields."""

    @abc.abstractmethod
    def _make_vf(
        self,
    ) -> torch.nn.Module:
        """Initializes the neural components of the velocity field."""

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
    ) -> VfFunction:
        """Compiles the velocity field function to be fed to external solvers."""


class MLPUnconditionalVF(BaseVelocityField):
    """Class for MLP-base unconditional neural velocity fields.

    The architecture of unconditional velocity fields is defined as follows:
        * (Optional) Time Featurization. An identity mapping is instantiated otherwise.
        * (Optional) Time MLP Encoder. An identity mapping is instantiated otherwise.
        * (Optional) State MLP Encoder. An identity mapping is instantiated otherwise.
        * Conditioning layer, whereby the state and time information are combined.
        * Velocity Field Decoder ultimately computing the velocity field.
    """

    def __init__(
        self,
        state_dim: int,
        encode_state: bool = True,
        encode_time: bool = True,
        time_features_id: TimeFeaturesId | None = None,
        time_features_fn: TTimeFeaturesFn | None = None,
        num_time_features: int | None = None,
        max_period: int | None = None,
        time_features_kwargs: dict[str, Any] | None = None,
        state_encoder_output_dim: int | None = None,
        time_encoder_output_dim: int | None = None,
        state_encoder_mlp_kwargs: dict[str, Any] | None = None,
        time_encoder_mlp_kwargs: dict[str, Any] | None = None,
        vf_decoder_mlp_kwargs: dict[str, Any] | None = None,
        conditioning_id: ConditioningLayersId | None = None,
        conditioning_fn: TConditioningFn | None = None,
        conditioning_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initializes the velocity field with the given settings.

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
            Ignored when :param: `conditioning_fn` is passed as well. When not provided, it will be set to
            :constant: `sc_flow._constants.DEFAULT_TIME_FEATURES_MAX_PERIOD`. Defaults to `None`.
        :type conditioning_id: class: `ConditioningLayersId`

        :param conditioning_fn: (Optional) Callable using for the instantiation of custom conditioning layers.
            When provided, takes precedence over string initialization. Defaults to `None`.
        :type conditioning_fn: class: `TConditioningFn`

        :param conditioning_kwargs: (Optional) Keyword arguments used to initialize the conditioning layer.
            Ignored when the using concatenation based conditioning. When setting :param: `conditioning_id`
            to `"resnet1d"`, it should match the signatures of the `__init__` method of the :class: `Resnet1d` class,
            raise :class: `TypeError` otherwise. Defaults to `None`.
        :type conditioning_kwargs: class: `dict[str, Any]`
        """
        super().__init__()
        self._state_dim = state_dim
        self._encode_state = encode_state
        self._encode_time = encode_time
        self._time_features_id = time_features_id
        self._time_features_fn = time_features_fn
        self._num_time_features = num_time_features
        self._max_period = DEFAULT_TIME_FEATURES_MAX_PERIOD if max_period is None else max_period
        self._time_features_kwargs = time_features_kwargs
        self._state_encoder_output_dim = (
            DEFAULT_VF_LATENT_STATE_DIM if state_encoder_output_dim is None else state_encoder_output_dim
        )
        self._time_encoder_output_dim = (
            DEFAULT_VF_LATENT_TIME_DIM if time_encoder_output_dim is None else time_encoder_output_dim
        )
        self._state_encoder_mlp_kwargs = {} if state_encoder_mlp_kwargs is None else state_encoder_mlp_kwargs
        self._time_encoder_mlp_kwargs = {} if time_encoder_mlp_kwargs is None else time_encoder_mlp_kwargs
        self._vf_decoder_mlp_kwargs = {} if vf_decoder_mlp_kwargs is None else vf_decoder_mlp_kwargs
        self._conditioning_id = conditioning_id
        self._conditioning_fn = conditioning_fn
        self._conditioning_kwargs = conditioning_kwargs

        self._vf = self._make_vf()

    @property
    def _use_time_features(
        self,
    ) -> bool:
        """Boolean flag indicating whether time featurization is used for the current velocity field.

        This is the case when either :param: `time_features_id` or :param: `time_features_fn`
        are passed during initialization.
        """
        return (self._time_features_id is not None) or (self._time_features_fn is not None)

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
            time_features_fn=self._time_features_fn,
            max_period=self._max_period,
            time_features_kwargs=self._time_features_kwargs,
        )
        return FunctionalModule(time_features_fn)

    def _make_time_encoder(
        self,
    ) -> BaseModule | torch.nn.Identity:
        """Initializes the optional time encoder.

        When :param: `encode_time` is set to `True` it will initialize a :class: `MLP` with the configurations
        specified in :param: `time_encoder_mlp_kwargs`. Returns an instance of :class: `torch.nn.Identity` otherwise.
        """
        if self._encode_time:
            verify_fn_kwargs_dictionary(MLP.__init__, self._time_encoder_mlp_kwargs)
            return MLP(
                self._get_num_time_features(),
                self._time_encoder_output_dim,
                **self._time_encoder_mlp_kwargs,
            )
        return torch.nn.Identity()

    def _make_state_encoder(
        self,
    ) -> BaseModule | torch.nn.Identity:
        """Initializes the optional state encoder.

        When :param: `encode_state` is set to `True` it will initialize a :class: `MLP` with the configurations
        specified in :param: `state_encoder_mlp_kwargs`. Returns an instance of :class: `torch.nn.Identity` otherwise.
        """
        if self._encode_state:
            verify_fn_kwargs_dictionary(MLP.__init__, self._state_encoder_mlp_kwargs)
            return MLP(
                self._state_dim,
                self._state_encoder_output_dim,
                **self._state_encoder_mlp_kwargs,
            )
        return torch.nn.Identity()

    def _make_conditioning_layer(
        self,
    ) -> BaseConditioningLayer:
        """Initializes the conditioning layer according to the configurations specified during initialization."""
        return get_conditioning_layer(
            self._state_encoder_output_dim if self._encode_state else self._state_dim,
            self._time_encoder_output_dim if self._encode_time else self._get_num_time_features(),
            latent_condition_dim=None,  # unconditional for the moment
            conditioning_id=self._conditioning_id,
            conditioning_fn=self._conditioning_fn,
            conditioning_kwargs=self._conditioning_kwargs,
        )

    def _make_vf_decoder(
        self,
        decoder_input_dim: int,
    ) -> BaseModule:
        """Initializes the velocity field decoder.

        It will initialize a :class: `MLP` with the configurations specified in :param: `vf_decoder_mlp_kwargs`.
        """
        verify_fn_kwargs_dictionary(MLP.__init__, self._vf_decoder_mlp_kwargs)
        return MLP(
            decoder_input_dim,
            self._state_dim,
            **self._vf_decoder_mlp_kwargs,
        )

    def _make_vf(
        self,
    ) -> torch.nn.Module:
        """Initializes the neural components of the velocity field.

        This is done by calling the `self._make_*` methods defined above.
        """
        modules = {
            "time_features": self._make_time_features(),
            "time_encoder": self._make_time_encoder(),
            "state_encoder": self._make_state_encoder(),
            "conditioning_layer": self._make_conditioning_layer(),
        }
        modules["vf_decoder"] = self._make_vf_decoder(
            modules["conditioning_layer"].output_dim,
        )
        return torch.nn.ModuleDict(modules)

    def forward(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Performs a forward computation pass on the neural velocity field.

        :param t: The current time index at which the velocity field is computed.
        :type t: class: `torch.Tensor`

        :param x: The current state at which the velocity field is computed.
        :type x: class: `torch.Tensor`
        """
        encoded_xt = self._vf["state_encoder"](x)

        encoded_t = self._vf["time_features"](t)
        encoded_t = self._vf["time_encoder"](encoded_t)

        encoded_concat = self._vf["conditioning_layer"](encoded_t, encoded_xt)
        return self._vf["vf_decoder"](encoded_concat)

    def get_vf_fn(
        self,
        *args,
        **kwargs,
    ) -> VfFunction:
        """Compiles the velocity field function to be fed to external solvers."""

        def _vf_fn(t: torch.Tensor, x: torch.Tensor):
            return self.forward(t, x, *args, **kwargs)

        return _vf_fn
