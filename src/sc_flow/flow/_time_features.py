import math
from collections.abc import Callable
from functools import partial

import torch

from sc_flow._constants import PI
from sc_flow._types import TimeFeaturesId
from sc_flow.flow._torch_utils import ensure_2d_tensor_with_singleton_trailing_dim

__all__ = [
    "get_time_features_fn",
    "log_sinusoidal_time_features",
    "sinusoidal_time_features",
]


def sinusoidal_time_features(
    t: torch.Tensor,
    num_time_features: int,
) -> torch.Tensor:
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
    if num_time_features % 2 != 0 or num_time_features <= 0:
        msg = "The number of time features should be an even positive integer."
        raise ValueError(msg)
    t = ensure_2d_tensor_with_singleton_trailing_dim(t)
    t = t * max_period
    freqs = torch.arange(num_time_features // 2, device=t.device) / (num_time_features // 2)
    freqs = -math.log(max_period) * freqs
    t = t * torch.exp(freqs)
    return torch.concatenate([torch.cos(t), torch.sin(t)], dim=-1)


def get_time_features_fn(
    num_time_features: int | None = None,
    time_features_id: TimeFeaturesId | None = None,
    max_period: int | None = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    if time_features_id is None:

        def _identity_time_features(t: torch.Tensor):
            return ensure_2d_tensor_with_singleton_trailing_dim(t)

        return _identity_time_features
    if time_features_id not in ("sinusoidal", "log-sinusoidal"):
        msg = (
            f"Time features identifier {time_features_id} is not supported."
            ' Possible options are `["sinusoidal", "log-sinusoidal"]`'
        )
        raise ValueError(msg)
    if num_time_features is None:
        raise ValueError(f"num_time_features is required for time_features_id={time_features_id!r} (no default).")
    if time_features_id == "sinusoidal":
        return partial(sinusoidal_time_features, num_time_features=num_time_features)
    if max_period is None:
        raise ValueError("max_period is required for time_features_id='log-sinusoidal' (no default).")
    return partial(log_sinusoidal_time_features, num_time_features=num_time_features, max_period=max_period)
