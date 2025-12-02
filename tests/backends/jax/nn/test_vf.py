from collections.abc import Callable
from typing import Any

import pytest
from jax import numpy as jnp
import jax

from sc_flow._types import TimeFeaturesId
from sc_flow.backends.jax._types import TTimeFeaturesFn
from sc_flow.backends.jax.nn._vf import (
    MLPUnconditionalVF,
)

batch_size = 32
n_samples = 50
state_dim = 20
num_time_features = 256
state_encoder_output_dim = 8
time_encoder_output_dim = 4


class TestVF:
    @pytest.mark.parametrize("encode_state", [True, False])
    @pytest.mark.parametrize("encode_time", [True, False])
    @pytest.mark.parametrize("time_features_id", ["ott-jax", "torch-cfm", None])
    @pytest.mark.parametrize("time_features_fn", [None])
    @pytest.mark.parametrize("num_time_features", [None, num_time_features])
    @pytest.mark.parametrize("state_encoder_output_dim", [state_encoder_output_dim])
    @pytest.mark.parametrize("time_encoder_output_dim", [time_encoder_output_dim])
    @pytest.mark.parametrize(
        "state_encoder_mlp_kwargs",
        [
            {
                "use_batchnorm": True,
                "use_layernorm": True,
                "batchnorm_bias": True,
            },
            {
                "use_batchnorm": True,
                "use_layernorm": False,
                "batchnorm_bias": True,
            },
            {
                "use_batchnorm": True,
                "use_layernorm": True,
                "batchnorm_bias": False,
            },
        ],
    )
    @pytest.mark.parametrize(
        "time_encoder_mlp_kwargs",
        [
            {
                "use_batchnorm": True,
                "use_layernorm": True,
                "batchnorm_bias": True,
            },
            {
                "use_batchnorm": True,
                "use_layernorm": False,
                "batchnorm_bias": True,
            },
            {
                "use_batchnorm": True,
                "use_layernorm": True,
                "batchnorm_bias": False,
            },
        ],
    )
    @pytest.mark.parametrize(
        "vf_decoder_mlp_kwargs",
        [
            {
                "use_batchnorm": True,
                "use_layernorm": True,
                "batchnorm_bias": True,
            },
            {
                "use_batchnorm": True,
                "use_layernorm": False,
                "batchnorm_bias": True,
            },
            {
                "use_batchnorm": True,
                "use_layernorm": True,
                "batchnorm_bias": False,
            },
        ],
    )
    @pytest.mark.parametrize("conditioning_id", [None, "concat", "resnet1d"])
    @pytest.mark.parametrize("conditioning_fn", [None])
    @pytest.mark.parametrize("conditioning_kwargs", [None])
    def test_vanilla_mlp_vf_forward(
        self,
        encode_state: bool,
        encode_time: bool,
        time_features_id: TimeFeaturesId | None,
        time_features_fn: TTimeFeaturesFn | None,
        num_time_features: int | None,
        state_encoder_output_dim: int,
        time_encoder_output_dim: int,
        state_encoder_mlp_kwargs: dict[str, Any],
        time_encoder_mlp_kwargs: dict[str, Any],
        vf_decoder_mlp_kwargs: dict[str, Any],
        conditioning_id: str | None,
        conditioning_fn: Callable | None,
        conditioning_kwargs: dict[str, Any],
    ):
        vf = MLPUnconditionalVF(
            state_dim,
            encode_state=encode_state,
            encode_time=encode_time,
            time_features_id=time_features_id,
            time_features_fn=time_features_fn,
            num_time_features=num_time_features,
            state_encoder_output_dim=state_encoder_output_dim,
            time_encoder_output_dim=time_encoder_output_dim,
            state_encoder_mlp_kwargs=state_encoder_mlp_kwargs,
            time_encoder_mlp_kwargs=time_encoder_mlp_kwargs,
            vf_decoder_mlp_kwargs=vf_decoder_mlp_kwargs,
            conditioning_id=conditioning_id,
            conditioning_fn=conditioning_fn,
            conditioning_kwargs=conditioning_kwargs,
        )

        rng_key = jax.random.PRNGKey(0)

        # case 0: x: (B, D) t: (B, 1)
        x = jnp.zeros((batch_size, state_dim))
        t = jnp.zeros((batch_size, 1))
        params = vf.init(rng_key, t, x)

        vt = vf.apply(params, t, x)
        assert vt.shape == (batch_size, state_dim)

        # case 1: x: (B, D) t: (B, )
        x = jnp.zeros((batch_size, state_dim))
        t = jnp.zeros((batch_size,))

        params = vf.init(rng_key, t, x)
        vt = vf.apply(params, t, x)
        assert vt.shape == (batch_size, state_dim)

        # case 2: x: (B, N, D) t: (B, 1)
        x = jnp.zeros((batch_size, n_samples, state_dim))
        t = jnp.zeros((batch_size, 1))

        params = vf.init(rng_key, t, x)
        vt = vf.apply(params, t, x)
        assert vt.shape == (batch_size, n_samples, state_dim)

        # case 2: x: (B, N, D) t: (B, )
        x = jnp.zeros((batch_size, n_samples, state_dim))
        t = jnp.zeros((batch_size,))

        params = vf.init(rng_key, t, x)
        vt = vf.apply(params, t, x)
        assert vt.shape == (batch_size, n_samples, state_dim)
