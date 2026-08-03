from collections.abc import Callable

import pytest
import torch

from sckitflow._constants import DEFAULT_NUM_TIME_FEATURES, DEFAULT_TIME_FEATURES_MAX_PERIOD
from sckitflow.core.nn._time_features import (
    get_time_features_fn,
    ott_jax_time_features,
    torch_cfm_time_features,
)

batch_size = 16
invalid_key = "invalid_key"


def valid_time_features_fn(
    t: torch.Tensor,
    num_time_features: int,
):
    return torch.ones((t.shape[0], num_time_features))


def invalid_time_features_fn(
    t: torch.Tensor,
    num_time_features: int,
):
    return torch.ones((t.shape[0], num_time_features + 1))


class TestTimeFeatures:
    @pytest.mark.parametrize("t", [torch.zeros(()), torch.zeros((batch_size, 1)), torch.zeros((batch_size,))])
    @pytest.mark.parametrize("num_time_features", [-1, 0, 256, 257])
    def test_ott_jax_time_features(
        self,
        t: torch.Tensor,
        num_time_features: int,
    ) -> None:
        if num_time_features % 2 != 0 or num_time_features <= 0:
            with pytest.raises(ValueError, match=r"The number of time features should be an even positive integer\."):
                t_features = ott_jax_time_features(t, num_time_features)
            return None
        t_features = ott_jax_time_features(t, num_time_features)
        if len(t.shape) == 0:
            assert t_features.shape == (1, num_time_features)
        else:
            assert t_features.shape == (batch_size, num_time_features)

    @pytest.mark.parametrize("t", [torch.zeros(()), torch.zeros((batch_size, 1)), torch.zeros((batch_size,))])
    @pytest.mark.parametrize("num_time_features", [-1, 0, 256, 257])
    def test_torch_cfm_time_features(
        self,
        t: torch.Tensor,
        num_time_features: int,
    ):
        if num_time_features % 2 != 0 or num_time_features <= 0:
            with pytest.raises(ValueError, match=r"The number of time features should be an even positive integer\."):
                t_features = torch_cfm_time_features(t, num_time_features, DEFAULT_TIME_FEATURES_MAX_PERIOD)
            return None
        t_features = torch_cfm_time_features(t, num_time_features, DEFAULT_TIME_FEATURES_MAX_PERIOD)
        if len(t.shape) == 0:
            assert t_features.shape == (1, num_time_features)
        else:
            assert t_features.shape == (batch_size, num_time_features)

    @pytest.mark.parametrize("t", [torch.zeros(()), torch.zeros((batch_size, 1)), torch.zeros((batch_size,))])
    @pytest.mark.parametrize("num_time_features", [None, 256])
    @pytest.mark.parametrize(
        "time_features_id",
        [
            None,
            "ott-jax",
            "torch-cfm",
            invalid_key,
        ],
    )
    @pytest.mark.parametrize(
        "time_features_fn",
        [
            None,
            valid_time_features_fn,
            invalid_time_features_fn,
        ],
    )
    @pytest.mark.parametrize("max_period", [None, DEFAULT_TIME_FEATURES_MAX_PERIOD])
    def test_get_time_features_fn(
        self,
        t: torch.Tensor,
        num_time_features: int | None,
        time_features_id: str | None,
        time_features_fn: Callable | None,
        max_period: int | None,
    ):
        if time_features_id == invalid_key:
            with pytest.raises(ValueError, match=r"Time features identifier .* is not supported\."):
                time_features = get_time_features_fn(
                    num_time_features=num_time_features,
                    time_features_id=time_features_id,
                    max_period=max_period,
                )
            return None

        time_features = get_time_features_fn(
            num_time_features=num_time_features,
            time_features_id=time_features_id,
            time_features_fn=time_features_fn,
            max_period=max_period,
        )
        if time_features_fn is not None and time_features_fn.__name__.startswith("invalid"):
            with pytest.raises(
                RuntimeError,
                match=(
                    r"Custom time features should return a tensor with shape .*\."
                    r"Found output with shape of .*\."
                ),
            ):
                t_features = time_features(t)
            return None
        t_features = time_features(t)
        if (time_features_id is not None) or (time_features_fn is not None):
            if num_time_features is None:
                num_time_features = DEFAULT_NUM_TIME_FEATURES
            if len(t.shape) == 0:
                assert t_features.shape == (1, num_time_features)
            else:
                assert t_features.shape == (batch_size, num_time_features)
        else:
            assert torch.equal(t_features.squeeze(), t.squeeze())
