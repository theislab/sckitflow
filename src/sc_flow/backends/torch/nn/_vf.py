import abc
from typing import Any

import torch

from sc_flow._constants import (
    DEFAULT_NUM_RESNET_LAYERS,
    DEFAULT_NUM_TIME_FEATURES,
    DEFAULT_VF_LATENT_STATE_DIM,
    DEFAULT_VF_LATENT_TIME_DIM,
    DEFAULT_TIME_FEATURES_MAX_PERIOD,
)
from sc_flow._types import TimeFeaturesId
from sc_flow._utils import verify_fn_kwargs_dictionary
from sc_flow.backends.torch._types import TTimeFeaturesFn, VfFunction
from sc_flow.backends.torch._utils import ensure_2d_tensor_with_singleton_trailing_dim, make_concatenation_possible
from sc_flow.backends.torch.nn._modules import MLP, Resnet1d
from sc_flow.backends.torch.nn._time_features import get_time_features_fn

__all__ = [
    "BaseVelocityField",
    "VanillaMLPVelocityField",
]


class BaseVelocityField(abc.ABC, torch.nn.Module):
    """"""  # noqa

    @property
    @abc.abstractmethod
    def _vf_decoder_input_dim(
        self,
    ) -> int:
        """"""  # noqa

    @abc.abstractmethod
    def _get_encoded_concat(
        self, encoded_t: torch.Tensor, encoded_state: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        """"""  # noqa

    @abc.abstractmethod
    def _make_vf(
        self,
    ) -> None:
        """"""  # noqa

    @abc.abstractmethod
    def get_vf_fn(
        self,
        *args,
        **kwargs,
    ) -> VfFunction:
        """"""  # noqa

    @abc.abstractmethod
    def forward(
        t: torch.Tensor,
        xt: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """"""  # noqa


class VanillaMLPVelocityField(BaseVelocityField):
    """"""  # noqa

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
    ) -> None:
        """"""  # noqa

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
        verify_fn_kwargs_dictionary(MLP.__init__, self._state_encoder_mlp_kwargs)
    
        self._time_encoder_mlp_kwargs = {} if time_encoder_mlp_kwargs is None else time_encoder_mlp_kwargs
        verify_fn_kwargs_dictionary(MLP.__init__, self._time_encoder_mlp_kwargs)
        
        self._vf_decoder_mlp_kwargs = {} if vf_decoder_mlp_kwargs is None else vf_decoder_mlp_kwargs
        verify_fn_kwargs_dictionary(MLP.__init__, self._vf_decoder_mlp_kwargs)

        self._make_vf()

    @property
    def _use_time_features(
        self,
    ) -> bool:
        """""" # noqa

        return (self._time_features_id is not None) or (self._time_features_fn is not None)

    @property
    def _vf_decoder_input_dim(
        self,
    ) -> int:
        """"""  # noqa

        if self._encode_state and self._encode_time:
            return self._state_encoder_output_dim + self._time_encoder_output_dim
        elif self._encode_state and (not self._encode_time):
            return self._state_encoder_output_dim + self._num_time_features
        elif (not self._encode_state) and self._encode_time:
            return self._state_dim + self._time_encoder_output_dim
        return self._state_dim + self._num_time_features

    def _make_vf(
        self,
    ) -> None:
        """"""  # noqa

        if not self._use_time_features:
            self._num_time_features = 1
            self._time_features = lambda t: ensure_2d_tensor_with_singleton_trailing_dim(t)

        else:
            if self._num_time_features is None:
                self._num_time_features = DEFAULT_NUM_TIME_FEATURES
            self._time_features = get_time_features_fn(
                num_time_features=self._num_time_features,
                time_features_id=self._time_features_id,
                time_features_fn=self._time_features_fn,
                max_period=self._max_period,
                time_features_kwargs=self._time_features_kwargs,
            )

        if self._encode_state:
            self._state_encoder = MLP(
                self._state_dim,
                self._state_encoder_output_dim,
                **self._state_encoder_mlp_kwargs,
            )

        if self._encode_time:
            self._time_encoder = MLP(
                self._num_time_features,
                self._time_encoder_output_dim,
                **self._time_encoder_mlp_kwargs,
            )

        self._vf_decoder = MLP(
            self._vf_decoder_input_dim,
            self._state_dim,
            **self._vf_decoder_mlp_kwargs,
        )

    def _get_encoded_concat(
        self,
        encoded_t: torch.Tensor,
        encoded_state: torch.Tensor,
    ) -> torch.Tensor:
        """"""  # noqa

        encoded_t = make_concatenation_possible(encoded_t, encoded_state, -1)
        return torch.concatenate((encoded_t, encoded_state), dim=-1)

    def forward(
        self,
        t: torch.Tensor,
        xt: torch.Tensor,
    ) -> torch.Tensor:
        """"""  # noqa

        original_shape = xt.shape
        if self._encode_state:
            encoded_state = self._state_encoder(xt)
        else:
            encoded_state = xt

        encoded_t = self._time_features(t)
        if self._encode_time:
            encoded_t = self._time_encoder(encoded_t)

        encoded_concat = self._get_encoded_concat(encoded_t, encoded_state)
        return self._vf_decoder(encoded_concat)

    def get_vf_fn(
        self,
        *args,
        **kwargs,
    ) -> VfFunction:
        """"""  # noqa

        def _vf_fn(t: torch.Tensor, xt: torch.Tensor):
            return self.forward(t, xt)

        return _vf_fn


class ResnetMLPVelocityField(VanillaMLPVelocityField):
    """"""  # noqa

    def __init__(
        self,
        state_dim: int,
        encode_state: bool = True,
        encode_time: bool = True,
        num_resnet_layers: int | None = None,
        time_features: TimeFeaturesId | TTimeFeaturesFn | None = None,
        time_features_kwargs: dict[str, Any] | None = None,
        num_time_features: int | None = None,
        state_encoder_output_dim: int | None = None,
        time_encoder_output_dim: int | None = None,
        state_encoder_mlp_kwargs: dict[str, Any] | None = None,
        time_encoder_mlp_kwargs: dict[str, Any] | None = None,
        vf_decoder_mlp_kwargs: dict[str, Any] | None = None,
        resnet_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """"""  # noqa

        self._num_resnet_layers = DEFAULT_NUM_RESNET_LAYERS if num_resnet_layers is None else num_resnet_layers
        self._resnet_kwargs = (
            {} if resnet_kwargs is None else verify_fn_kwargs_dictionary(Resnet1d, "__init__", resnet_kwargs)
        )

        super().__init__(
            state_dim,
            encode_state=encode_state,
            encode_time=encode_time,
            time_features=time_features,
            time_features_kwargs=time_features_kwargs,
            num_time_features=num_time_features,
            state_encoder_output_dim=state_encoder_output_dim,
            time_encoder_output_dim=time_encoder_output_dim,
            state_encoder_mlp_kwargs=state_encoder_mlp_kwargs,
            time_encoder_mlp_kwargs=time_encoder_mlp_kwargs,
            vf_decoder_mlp_kwargs=vf_decoder_mlp_kwargs,
        )

    @property
    def _vf_decoder_input_dim(
        self,
    ) -> int:
        """"""  # noqa

        return self._state_encoder_output_dim

    def _make_vf(
        self,
    ) -> torch.nn.Module:
        """"""  # noqa

        super()._make_vf()
        self._resnet = Resnet1d(
            self._state_encoder_output_dim,
            self._num_resnet_layers,
            **self._resnet_kwargs,
        )

    def _get_encoded_concat(
        self,
        encoded_t: torch.Tensor,
        encoded_state: torch.Tensor,
    ) -> torch.Tensor:
        """"""  # noqa

        encoded_t = make_concatenation_possible(encoded_t, encoded_state, -1)
        return self._resnet(encoded_state, encoded_t)
