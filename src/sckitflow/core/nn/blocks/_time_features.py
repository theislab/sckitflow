import math
from collections.abc import Callable
from functools import partial
from typing import ClassVar, Protocol, overload

import torch
from scfit.registry import Component, component

from sckitflow._constants import PI
from sckitflow.core._utils import ensure_2d_tensor_with_singleton_trailing_dim

__all__ = [
    "TimeFeaturesFn",
    "TimeFeaturesConfig",
    "OTTJaxTimeFeaturesConfig",
    "TorchCFMTimeFeaturesConfig",
]


def _ott_jax_time_features(
    t: torch.Tensor,
    n_time_features: int,
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

    :param n_time_features: Sets the value of $2K$, the number of resulting time features, hence it must be even.
        Raises a :class: `ValueError` otherwise.
    :type n_time_features: class: `int`
    """
    if n_time_features % 2 != 0 or n_time_features <= 0:
        msg = "The number of time features should be an even positive integer."
        raise ValueError(msg)
    t = ensure_2d_tensor_with_singleton_trailing_dim(t)
    freq = 2 * torch.arange(n_time_features // 2, device=t.device) * PI
    t = t * freq
    return torch.concatenate([torch.cos(t), torch.sin(t)], dim=-1)


def _torch_cfm_time_features(
    t: torch.Tensor,
    n_time_features: int,
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

    :param n_time_features: Sets the value of $2K$, the number of resulting time features, hence it must be even.
        Raises a :class: `ValueError` otherwise.
    :type n_time_features: class: `int`

    :param max_period: Sets the value of $M$, used for the linear scaling of the time features.
    :type max_period: class: `int`
    """
    if n_time_features % 2 != 0 or n_time_features <= 0:
        msg = "The number of time features should be an even positive integer."
        raise ValueError(msg)
    t = ensure_2d_tensor_with_singleton_trailing_dim(t)
    freqs = torch.arange(n_time_features // 2, device=t.device) / (n_time_features // 2)
    freqs = -math.log(max_period) * freqs
    t = t * torch.exp(freqs)
    return torch.concatenate([torch.cos(t), torch.sin(t)], dim=-1)


class TimeFeaturesFn(Protocol):
    @overload
    def __call__(self, t: torch.Tensor, n_time_features: int): ...
    @overload
    def __call__(self, t: torch.Tensor, n_time_features: int, max_period: int): ...


class TimeFeaturesConfig(Component):
    fn: ClassVar[TimeFeaturesFn]
    n_time_features: int

    def build(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return partial(type(self).fn, **dict(self))


@component("nn.time_features.ott_jax")
class OTTJaxTimeFeaturesConfig(TimeFeaturesConfig):
    fn: ClassVar[TimeFeaturesFn] = _ott_jax_time_features


@component("nn.time_features.torch_cfm")
class TorchCFMTimeFeaturesConfig(TimeFeaturesConfig):
    fn: ClassVar[TimeFeaturesFn] = _torch_cfm_time_features
    max_period: int
