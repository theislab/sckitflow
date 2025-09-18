import math
from collections.abc import Callable
from functools import partial
from typing import Any

import torch

from sc_flow._constants import DEFAULT_NUM_TIME_FEATURES, DEFAULT_TIME_FEATURES_MAX_PERIOD, PI
from sc_flow._types import TimeFeaturesId
from sc_flow._utils import verify_fn_signature
from sc_flow.backends.torch._types import TTimeFeaturesFn
from sc_flow.backends.torch._utils import ensure_2d_tensor_with_singleton_trailing_dim

__all__ = [
    "ott_jax_time_features",
    "torch_cfm_time_features",
    "make_custom_time_features",
    "get_time_features_fn",
]


def ott_jax_time_features(
    t: torch.Tensor,
    num_time_features: int,
) -> torch.Tensor:
    r"""Applies the sinusoidal time features on the current time index as defined in the `ott-jax`package.

    You can find the original implementation at the following link
    https://github.com/ott-jax/ott/blob/main/src/ott/neural/networks/layers/time_encoder.py.
    For integer $K\in\mathbb{N}$, the featurized representation $\tilde{t}_K$ of time index vector $t$ is computed as:

    $$
        \tilde{t}_K:=[(\cos(2\pi i t), \sin(2\pi i t))]_{i=1}^K
    $$

    Hence resulting in $2K$ features that are then fed to the Neural Velocity Field.

    :param t: The current continuous time index $t$ on which the features $\tilde{t}_K$ are computed.
        Must be a :class: `torch.Tensor` of shape $()$, $(B, )$ or $(B, 1)$, where $B$ denotes the batch size.
    :type t: class: `torch.Tensor`

    :param num_time_features: Sets the value of $2K$, the number of resulting time features, hence it must be even.
        Raises a :class: `ValueError` otherwise.
    :type num_time_features: class: `int`
    """
    if num_time_features % 2 != 0 or num_time_features <= 0:
        msg = "The number of time features should be an even positive integer."
        raise ValueError(msg)
    t = ensure_2d_tensor_with_singleton_trailing_dim(t)
    freq = 2 * torch.arange(num_time_features // 2, device=t.device) * PI
    t = t * freq
    return torch.concatenate([torch.cos(t), torch.sin(t)], dim=-1)


def torch_cfm_time_features(
    t: torch.Tensor,
    num_time_features: int,
    max_period: int,
) -> torch.Tensor:
    r"""Applies the sinusoidal time features on the current time index as defined in the `ott-jax`package.

    You can find the original implementation at the following link
    https://github.com/atong01/conditional-flow-matching/blob/66b236b2fd96466a9aa655f46df2b4762aa11281/torchcfm/models/unet/nn.py#L87
    For integers $K, N\in\mathbb{M}$, the featurized representation $\tilde{t}_K$ of time index vector $t$ is computed as:

    $$
        - t \exp(\frac{\pi i }{K} \log M)
        \tilde{t}_K:=[(\cos(- t \exp(\frac{i}{K} \log M)), \sin(- t \exp(\frac{i}{K} \log M)))]_{i=1}^K
    $$

    Hence resulting in $2K$ features that are then fed to the Neural Velocity Field.

    :param t: The current continuous time index $t$ on which the features $\tilde{t}_K$ are computed.
        Must be a :class: `torch.Tensor` of shape $()$, $(B, )$ or $(B, 1)$, where $B$ denotes the batch size.
    :type t: class: `torch.Tensor`

    :param num_time_features: Sets the value of $2K$, the number of resulting time features, hence it must be even.
        Raises a :class: `ValueError` otherwise.
    :type num_time_features: class: `int`

    :param max_period: Sets the value of $M$, used for the linear scaling of the time features.
    :type max_period: class: `int`
    """
    if num_time_features % 2 != 0 or num_time_features <= 0:
        msg = "The number of time features should be an even positive integer."
        raise ValueError(msg)
    t = ensure_2d_tensor_with_singleton_trailing_dim(t)
    freqs = torch.arange(num_time_features // 2, device=t.device) / (num_time_features // 2)
    freqs = -math.log(max_period) * freqs
    t = t * torch.exp(freqs)
    return torch.concatenate([torch.cos(t), torch.sin(t)], dim=-1)


def make_custom_time_features(
    time_features_fn: TTimeFeaturesFn, num_time_features: int, time_features_kwargs: dict[str, Any]
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Compiles the input function with the chosen arguments and return its wrapped version.

    :param time_features_fn: (Optional) Function defining the custom time featurizer which should accept tensor with
            trailing singleton dimension and expand the trailing dimension to $2K$.The input function is wrapped,
            so that it only need to actually implement the expansion of the trailing dimension. Defaults to `None`.
    :type time_features_fn: class: `TTimeFeaturesFn | None`

    :param num_time_features: (Optional) Sets the value of $2K$, the number of resulting time features, hence it must be even.
        Raises a :class: `ValueError` otherwise. When not provided, it will be set to
        :constant: `sc_flow._constants.DEFAULT_NUM_TIME_FEATURES`. Defaults to `None`.
    :type num_time_features: class: `int | None`

    :param time_features_kwargs: (Optional) Keyword arguments used to compile custom time featurizers. Only used when
        initializing time featurizers with :param: `time_features_fn`, ignored otherwise. Defaults to `None`
    :type time_features_kwargs: class: `dict[str, Any] | None`
    """
    if num_time_features <= 0:
        msg = "The number of time features should be positive."
        raise ValueError(msg)
    verify_fn_signature(time_features_fn, TTimeFeaturesFn, kwargs_dict=time_features_kwargs)

    def _time_features_fn(t: torch.Tensor):
        t = ensure_2d_tensor_with_singleton_trailing_dim(t)
        t_features = time_features_fn(t, num_time_features=num_time_features, **time_features_kwargs)
        if t_features.shape != (t.shape[0], num_time_features):
            msg = (
                f"Custom time features should return a tensor with shape {(t.shape[0], num_time_features)}."
                f"Found output with shape of {t_features.shape}."
            )
            raise RuntimeError(msg)
        return t_features

    return _time_features_fn


def get_time_features_fn(
    num_time_features: int | None = None,
    time_features_id: TimeFeaturesId | None = None,
    time_features_fn: TTimeFeaturesFn | None = None,
    max_period: int | None = None,
    time_features_kwargs: dict[str, Any] | None = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Compiles a time feature function with the given arguments and returns it.

    It supports initialization of time features functions in the following two ways:

        1) Pre-defined time time featurization functions can be chosen by passing the
            :param: `time_features_id` argument, which can be set to either `"ott-jax"`
            or `"torch-cfm"`.
        2) It is otherwise possible to initialize custom time featurizers by passing
            the :param: `time_features_fn`, which needs to be a :class: `Callable[torch.Tensor, int, ...]`
            where the first argument correspond to $t$ and the second one should be named `num_time_features`
            and represent $2K$.

    When arguments for both method 1 and 2 are passed, the latter will take precedence.
    When no argument is provided, returns the identity function. As all the defaults arguments are set to `None`, this will
    be the default case when calling this function without any argument (i.e.: calling `get_time_features_fn()`).

    :param num_time_features: (Optional) Sets the value of $2K$, the number of resulting time features, hence it must be even.
        Raises a :class: `ValueError` otherwise. When not provided, it will be set to
        :constant: `sc_flow._constants.DEFAULT_NUM_TIME_FEATURES`. Defaults to `None`.
    :type num_time_features: class: `int | None`

    :param time_features_id: (Optional) String identifier indicating which of the predefined time features to retrieve.
        Could be set to either `"ott-jax"` or `"torch-cfm"`. Raises :class: `ValueError` when a different identifier is specified.
        Defaults to `None`.
    :class time_features_id: class:`TimeFeaturesId | None`

    :param time_features_fn: (Optional) Function defining the custom time featurizer which should accept tensor with
            trailing singleton dimension and expand the trailing dimension to $2K$.The input function is wrapped,
            so that it only need to actually implement the expansion of the trailing dimension. Defaults to `None`.
    :type time_features_fn: class: `TTimeFeaturesFn | None`

    :param max_period: Sets the value of $M$, used for the linear scaling of the time features.
        Only used when :param: `time_features_id` is set to `"torch-cfm"`, ignored otherwise.
        When not provided, it will be set to :constant: `sc_flow._constants.DEFAULT_TIME_FEATURES_MAX_PERIOD`. Defaults to `None`.
    :type max_period: class: `int | None`

    :param time_features_kwargs: (Optional) Keyword arguments used to compile custom time featurizers. Only used when
        initializing time featurizers with :param: `time_features_fn`, ignored otherwise. Defaults to `None`
    :type time_features_kwargs: class: `dict[str, Any] | None`
    """
    if time_features_fn is not None:
        num_time_features = DEFAULT_NUM_TIME_FEATURES if num_time_features is None else num_time_features
        time_features_kwargs = {} if time_features_kwargs is None else time_features_kwargs
        return make_custom_time_features(time_features_fn, num_time_features, time_features_kwargs)
    if time_features_id is None:

        def _time_features_fn(t: torch.Tensor):
            return ensure_2d_tensor_with_singleton_trailing_dim(t)

        return _time_features_fn
    if time_features_id == "ott-jax":
        num_time_features = DEFAULT_NUM_TIME_FEATURES if num_time_features is None else num_time_features
        return partial(ott_jax_time_features, num_time_features=num_time_features)
    if time_features_id == "torch-cfm":
        num_time_features = DEFAULT_NUM_TIME_FEATURES if num_time_features is None else num_time_features
        max_period = DEFAULT_TIME_FEATURES_MAX_PERIOD if max_period is None else max_period
        return partial(torch_cfm_time_features, num_time_features=num_time_features, max_period=max_period)
    else:
        msg = (
            f"Time features identifier {time_features_id} is not supported."
            ' Possible options are `["ott-jax", "torch-cfm"]`'
        )
        raise ValueError(msg)
