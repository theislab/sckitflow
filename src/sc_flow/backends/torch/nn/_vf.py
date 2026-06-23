import abc
from collections.abc import Collection
from typing import Any, Literal

import torch

from sc_flow._constants import (
    DEFAULT_CONDITION_ENCODER_OUTPUT_DIM,
    DEFAULT_NUM_TIME_FEATURES,
    DEFAULT_SOURCE_ENCODER_OUTPUT_DIM,
    DEFAULT_TIME_FEATURES_MAX_PERIOD,
    DEFAULT_VF_LATENT_STATE_DIM,
    DEFAULT_VF_LATENT_TIME_DIM,
)
from sc_flow._types import ConditioningLayersId, LayersDict, NestedLayersDict, TimeFeaturesId
from sc_flow.backends.torch._types import MappedTensor, TConditioningFn, TTimeFeaturesFn, TVfFn
from sc_flow.backends.torch._utils import make_concatenation_possible
from sc_flow.backends.torch.nn._conditioning_layers import BaseConditioningLayer, get_conditioning_layer
from sc_flow.backends.torch.nn._modules import BaseModule, FunctionalModule
from sc_flow.backends.torch.nn._set_encoder import SetEncoder
from sc_flow.backends.torch.nn._time_features import get_time_features_fn
from sc_flow.backends.torch.nn._utils import init_module_from_dict
from sc_flow.data._dims_registry import DataDimensionalitiesRegistry

__all__ = [
    "BaseVelocityField",
    "MLPVelocity",
]


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
        state_encoder_mlp_kwargs: LayersDict | None = None,
        time_encoder_mlp_kwargs: LayersDict | None = None,
        vf_decoder_mlp_kwargs: LayersDict | None = None,
        conditioning_id: ConditioningLayersId | None = None,
        conditioning_fn: TConditioningFn | None = None,
        conditioning_kwargs: dict[str, Any] | None = None,
        condition_encoder_input_layers: NestedLayersDict | None = None,
        condition_encoder_output_dim: int | None = None,
        condition_encoder_pooling_mode: Literal["mean", "sum"] = "mean",
        condition_encoder_pooling_kwargs: dict[str, Any] | None = None,
        condition_encoder_pooling_proj_dim: int | None = None,
        condition_encoder_pooling_proj_bias: bool = True,
        condition_encoder_covariates_not_pooled: Collection[str] | None = None,
        condition_encoder_output_layers_kwargs: LayersDict | None = None,
        source_encoder_mlp_kwargs: LayersDict | None = None,
        source_encoder_output_dim: int | None = None,
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

        :param condition_encoder_pooling_proj_dim: Shared projection dimension for the covariates to pool,
            defaults to `None`, in which case it will be set to the minimum output
            dimensionality of the input encoder for pooled covariates.
        :type condition_encoder_pooling_proj_dim: class: `int | None`

        :param condition_encoder_pooling_proj_bias: Whether to use bias term for linear projection of
            covariates to pool, defaults to `True`.
        :type condition_encoder_pooling_proj_bias: class: `bool`

        :param condition_encoder_covariates_not_pooled: Collection of string identifiers for the covariates not to pool,
            defaults to `None`.
        :type condition_encoder_covariates_not_pooled: class: `Collection[str] | None`

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
        self._conditioning_kwargs = {} if conditioning_kwargs is None else conditioning_kwargs
        self._condition_encoder_input_layers = condition_encoder_input_layers
        self._condition_encoder_output_dim = (
            DEFAULT_CONDITION_ENCODER_OUTPUT_DIM
            if condition_encoder_output_dim is None
            else condition_encoder_output_dim
        )
        self._condition_encoder_pooling_mode = condition_encoder_pooling_mode
        self._condition_encoder_pooling_kwargs = condition_encoder_pooling_kwargs
        self._condition_encoder_pooling_proj_dim = condition_encoder_pooling_proj_dim
        self._condition_encoder_pooling_proj_bias = condition_encoder_pooling_proj_bias
        self._condition_encoder_covariates_not_pooled = condition_encoder_covariates_not_pooled
        self._condition_encoder_output_layers_kwargs = condition_encoder_output_layers_kwargs
        self._source_encoder_mlp_kwargs = source_encoder_mlp_kwargs
        self._source_encoder_output_dim = (
            DEFAULT_SOURCE_ENCODER_OUTPUT_DIM if source_encoder_output_dim is None else source_encoder_output_dim
        )

        self._vf = self._make_modules()

    @property
    def _use_time_features(
        self,
    ) -> bool:
        """Boolean flag indicating whether time featurization is used for the current velocity field.

        This is the case when either :param: `time_features_id` or :param: `time_features_fn`
        are passed during initialization.
        """
        return (self._time_features_id is not None) or (self._time_features_fn is not None)

    @property
    def use_source_encoder(
        self,
    ) -> bool:
        """Boolean flag indicating whether an MLP should be initialized to encode source states."""
        return self._source_encoder_mlp_kwargs is not None

    @property
    def _source_encoder_input_dim(self) -> int:
        """Returns the input dimensionality for the source encoder.

        When not explicitly specified in :attr: `self._source_encoder_mlp_kwargs`, it will be
        automatically retrieved from :attr: `self._state_dim`
        """
        input_dim = self._source_encoder_mlp_kwargs.get("input_dim", None)
        if input_dim is None:
            return self._state_dim
        return input_dim

    @property
    def _conditioning_dim(
        self,
    ) -> int | None:
        """Returns the dimensionality for the condition input of the conditioning layers."""
        if self.is_conditional and self.use_source_encoder:
            return self._condition_encoder_output_dim + self._source_encoder_output_dim
        elif self.is_conditional and (not self.use_source_encoder):
            return self._condition_encoder_output_dim
        elif self.use_source_encoder:
            return self._source_encoder_output_dim
        return None

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

    def _get_encoded_conditions(
        self,
        condition_dict: MappedTensor | None,
    ) -> torch.Tensor | None:
        """Retrieves the encoded conditions."""
        if self.is_conditional:
            if condition_dict is None:
                msg = "Conditional VFs should take a condition as input, found `None`."
                raise TypeError(msg)
            return self._vf["condition_encoder"](condition_dict)
        return None

    def _get_encoded_source(
        self,
        source: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Retrieves the encoded source states."""
        if self.use_source_encoder:
            if source is None:
                msg = "When using source encoder a source state should be passed, found `None`."
                raise TypeError(msg)
            return self._vf["source_encoder"](source)
        return None

    def _get_conditioning_input(
        self, encoded_condition: torch.Tensor, encoded_source: torch.tensor
    ) -> torch.Tensor | None:
        """Retrieves the condition input for the conditioning layers."""
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
            return init_module_from_dict(
                self._time_encoder_mlp_kwargs,
                input_dim=self._get_num_time_features(),
                output_dim=self._time_encoder_output_dim,
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
            return init_module_from_dict(
                self._state_encoder_mlp_kwargs,
                input_dim=self._state_dim,
                output_dim=self._state_encoder_output_dim,
            )
        return torch.nn.Identity()

    def _make_conditioning_layer(
        self,
    ) -> BaseConditioningLayer:
        """Initializes the conditioning layer according to the configurations specified during initialization."""
        return get_conditioning_layer(
            self._state_encoder_output_dim if self._encode_state else self._state_dim,
            self._time_encoder_output_dim if self._encode_time else self._get_num_time_features(),
            latent_condition_dim=self._conditioning_dim,
            conditioning_id=self._conditioning_id,
            conditioning_fn=self._conditioning_fn,
            conditioning_kwargs=self._conditioning_kwargs,
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
            self._condition_encoder_input_layers,
            self._condition_encoder_output_dim,
            pooling_mode=self._condition_encoder_pooling_mode,
            pooling_kwargs=self._condition_encoder_pooling_kwargs,
            pooling_proj_dim=self._condition_encoder_pooling_proj_dim,
            pooling_proj_bias=self._condition_encoder_pooling_proj_bias,
            covariates_not_pooled=self._condition_encoder_covariates_not_pooled,
            output_layers_kwargs=self._condition_encoder_output_layers_kwargs,
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
            self._source_encoder_mlp_kwargs,
            input_dim=self._source_encoder_input_dim,
            output_dim=self._source_encoder_output_dim,
        )

    def _make_modules(
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
        if self.is_conditional:
            modules["condition_encoder"] = self._make_condition_encoder()
        if self.use_source_encoder:
            modules["source_encoder"] = self._make_source_encoder()
        modules["vf_decoder"] = self._make_vf_decoder(
            modules["conditioning_layer"].output_dim,
        )
        return torch.nn.ModuleDict(modules)

    def forward(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        condition_dict: MappedTensor | None = None,
        source: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Performs a forward computation pass on the neural velocity field.

        :param t: The current time index at which the velocity field is computed.
        :type t: class: `torch.Tensor`

        :param x: The current state at which the velocity field is computed.
        :type x: class: `torch.Tensor`

        :param condition_dict: The input dictionary containing the data for
            each perturbation covariate.
        :type condition_dict: class: `MappedTensor`
        """
        encoded_xt = self._vf["state_encoder"](x)

        encoded_t = self._vf["time_features"](t)
        encoded_t = self._vf["time_encoder"](encoded_t)

        encoded_condition = self._get_encoded_conditions(condition_dict)
        encoded_source = self._get_encoded_source(source)

        encoded_concat = self._vf["conditioning_layer"](
            encoded_t, encoded_xt, encoded_condition=self._get_conditioning_input(encoded_condition, encoded_source)
        )
        return self._vf["vf_decoder"](encoded_concat)

    def get_vf_fn(
        self,
        condition_dict: MappedTensor | None = None,
        source: torch.Tensor | None = None,
    ) -> TVfFn:
        """Compiles the velocity field function to be fed to external solvers."""

        def _vf_fn(t: torch.Tensor, x: torch.Tensor):
            return self.forward(t, x, condition_dict=condition_dict, source=source)

        return _vf_fn

    @property
    def is_conditional(
        self,
    ) -> bool:
        """Whether a condition encoder is associated to velocity field."""
        return self._condition_encoder_input_layers is not None

    @classmethod
    def _use_source_encoder(cls, is_paired_setting: bool, generate_from_noise: bool) -> bool:
        """Returns a boolean flag indicating whether to initialize the source encore module.

        The source encoder module will be initialized only in the paired settings when generating from noise.

        :param is_paired_setting: Boolean flag indicating whether the data is configured in the paired setting.
        :type is_paired_setting: class: `bool`

        :param generate_from_noise: Boolean flag indicating whether the interpolation is made between tractable
            noise distribution and data.
        :type generate_from_noise: class: `bool`
        """
        if is_paired_setting and generate_from_noise:
            return True
        return False

    @classmethod
    def init_from_dims_registry(
        cls,
        dims_registry: DataDimensionalitiesRegistry,
        is_paired_setting: bool,
        generate_from_noise: bool = False,
        condition_encoder_input_layers: NestedLayersDict | None = None,
        source_encoder_mlp_kwargs: LayersDict | None = None,
        **kwargs,
    ) -> "MLPVelocity":
        # get dimensionalities from registry
        state_dim = dims_registry.state_dim

        # get covariates not to pool from registry
        condition_encoder_covariates_not_pooled = []

        # create dictionary with all conditions dimensions
        all_dims_dict = {
            **dims_registry.condition_reps_dims,
            **dims_registry.condition_continuous_dims,
            **dims_registry.groups_reps_dims,
        }

        # register input dimensionalities for condition encoder
        if condition_encoder_input_layers is not None:
            for cov, input_layers in condition_encoder_input_layers.items():
                # check that covariate appears in the data
                if cov not in all_dims_dict.keys():
                    msg = f"Covariate {cov} not found in the data."
                    raise KeyError(msg)

                # update dimensionality
                cov_input_dim = all_dims_dict[cov]
                input_layers["input_dim"] = cov_input_dim

                # add continuous covariates to the covariates not to pool
                if cov in dims_registry.condition_continuous_dims.keys():
                    condition_encoder_covariates_not_pooled.append(cov)

        # register source state dimensionality when provided
        if cls._use_source_encoder(is_paired_setting, generate_from_noise):
            if source_encoder_mlp_kwargs is None:
                source_encoder_mlp_kwargs = {}
            source_dim = dims_registry.source_dim
            source_encoder_mlp_kwargs["input_dim"] = source_dim

        # promote arguments passed from kwargs
        condition_encoder_input_layers = kwargs.pop("condition_encoder_input_layers", condition_encoder_input_layers)
        source_encoder_mlp_kwargs = kwargs.pop("source_encoder_mlp_kwargs", source_encoder_mlp_kwargs)
        condition_encoder_covariates_not_pooled = kwargs.pop(
            "condition_encoder_covariates_not_pooled", condition_encoder_covariates_not_pooled
        )

        return cls(
            state_dim,
            condition_encoder_input_layers=condition_encoder_input_layers,
            source_encoder_mlp_kwargs=source_encoder_mlp_kwargs,
            condition_encoder_covariates_not_pooled=condition_encoder_covariates_not_pooled,
            **kwargs,
        )
