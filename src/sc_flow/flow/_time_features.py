import math
from collections.abc import Callable
from functools import partial

import torch

from sc_flow._constants import DEFAULT_NUM_TIME_FEATURES, DEFAULT_TIME_FEATURES_MAX_PERIOD, PI
from sc_flow._types import TimeFeaturesId
from sc_flow.core._torch_utils import ensure_2d_tensor_with_singleton_trailing_dim

__all__ = [
    "get_time_features_fn",
    "log_sinusoidal_time_features",
    "sinusoidal_time_features",
]


def sinusoidal_time_features(
    t: torch.Tensor,
    num_time_features: int,
) -> torch.Tensor:
    r"""Sinusoidal time features with **linear** frequencies.

    For integer $K\in\mathbb{N}$, the featurized representation $\tilde{t}_K$ of time index vector $t$ is:

    $$
        \tilde{t}_K:=[(\cos(2\pi i t), \sin(2\pi i t))]_{i=1}^K
    $$

    Hence resulting in $2K$ features that are then fed to the Neural Velocity Field.

    Equivalent to the sinusoidal time embedding used in ``ott-jax``
    (``ott/neural/networks/layers/time_encoder.py``).

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


def log_sinusoidal_time_features(
    t: torch.Tensor,
    num_time_features: int,
    max_period: int,
) -> torch.Tensor:
    r"""Sinusoidal time features with **log-spaced** frequencies (the standard transformer positional embedding).

    For integers $K, M\in\mathbb{N}$, the featurized representation $\tilde{t}_K$ of time index vector $t$ is:

    $$
        \tilde{t}_K:=[(\cos(t \exp(-\tfrac{i}{K}\log M)), \sin(t \exp(-\tfrac{i}{K}\log M)))]_{i=1}^K
    $$

    Hence resulting in $2K$ features that are then fed to the Neural Velocity Field. ``max_period`` ($M$) sets
    the lowest frequency / longest period represented.

    Equivalent to the timestep embedding used in ``torchcfm``
    (``torchcfm/models/unet/nn.py``).

    :param t: The current continuous time index $t$ on which the features $\tilde{t}_K$ are computed.
        Must be a :class: `torch.Tensor` of shape $()$, $(B, )$ or $(B, 1)$, where $B$ denotes the batch size.
    :type t: class: `torch.Tensor`

    :param num_time_features: Sets the value of $2K$, the number of resulting time features, hence it must be even.
        Raises a :class: `ValueError` otherwise.
    :type num_time_features: class: `int`

    :param max_period: Sets the value of $M$, used for the log scaling of the time features.
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


def get_time_features_fn(
    num_time_features: int | None = None,
    time_features_id: TimeFeaturesId | None = None,
    max_period: int | None = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Compiles a time-features function from a flat identifier.

    ``time_features_id`` selects a predefined featurizer:

        * ``"sinusoidal"``     — linear-frequency sinusoidal features (:func:`sinusoidal_time_features`).
        * ``"log-sinusoidal"`` — log-spaced sinusoidal features (:func:`log_sinusoidal_time_features`),
          which additionally uses ``max_period``.

    When ``time_features_id`` is ``None`` (the default) the identity featurizer is returned — the scalar
    time is passed through as a trailing-singleton feature.

    :param num_time_features: (Optional) Sets the value of $2K$, the number of resulting time features, hence it must be even.
        Raises a :class: `ValueError` otherwise. When not provided, it will be set to
        :constant: `sc_flow._constants.DEFAULT_NUM_TIME_FEATURES`. Defaults to `None`.
    :type num_time_features: class: `int | None`

    :param time_features_id: (Optional) String identifier indicating which of the predefined time features to retrieve.
        Could be set to either `"sinusoidal"` or `"log-sinusoidal"`. Raises :class: `ValueError` when a different
        identifier is specified. Defaults to `None`.
    :type time_features_id: class:`TimeFeaturesId | None`

    :param max_period: Sets the value of $M$, used for the log scaling of the time features.
        Only used when :param: `time_features_id` is set to `"log-sinusoidal"`, ignored otherwise.
        When not provided, it will be set to :constant: `sc_flow._constants.DEFAULT_TIME_FEATURES_MAX_PERIOD`. Defaults to `None`.
    :type max_period: class: `int | None`
    """
    if time_features_id is None:

        def _identity_time_features(t: torch.Tensor):
            return ensure_2d_tensor_with_singleton_trailing_dim(t)

        return _identity_time_features
    if time_features_id == "sinusoidal":
        num_time_features = DEFAULT_NUM_TIME_FEATURES if num_time_features is None else num_time_features
        return partial(sinusoidal_time_features, num_time_features=num_time_features)
    if time_features_id == "log-sinusoidal":
        num_time_features = DEFAULT_NUM_TIME_FEATURES if num_time_features is None else num_time_features
        max_period = DEFAULT_TIME_FEATURES_MAX_PERIOD if max_period is None else max_period
        return partial(log_sinusoidal_time_features, num_time_features=num_time_features, max_period=max_period)
    msg = (
        f"Time features identifier {time_features_id} is not supported."
        ' Possible options are `["sinusoidal", "log-sinusoidal"]`'
    )
    raise ValueError(msg)
