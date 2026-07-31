from typing import Any

import pytest
import torch

from sckitflow.backends.torch.nn._vf import (
    MLPVelocity,
)

batch_size = 32
n_samples = 50
state_dim = 20
num_time_features = 256
state_encoder_output_dim = 8
time_encoder_output_dim = 4

# `encode_state`, `encode_time`, the time featurizer and the two encoder output dims
# genuinely interact (an output dim only matters when its encoder is enabled, and the
# featurizer feeds the time encoder), but the full 2*2*3*2*2*2 product is mostly
# redundant. These 12 rows are a pairwise cover: every pair of values across any two
# of the six axes appears at least once.
ENCODER_CONFIGS = [
    pytest.param(
        {"encode_state": True, "encode_time": True, "time_features_id": None},
        id="enc-both",
    ),
    pytest.param(
        {
            "encode_state": True,
            "encode_time": True,
            "time_features_id": "ott-jax",
            "num_time_features": num_time_features,
            "state_encoder_output_dim": state_encoder_output_dim,
            "time_encoder_output_dim": time_encoder_output_dim,
        },
        id="enc-both-ottjax-dims",
    ),
    pytest.param(
        {
            "encode_state": True,
            "encode_time": True,
            "time_features_id": "torch-cfm",
            "state_encoder_output_dim": state_encoder_output_dim,
        },
        id="enc-both-cfm-statedim",
    ),
    pytest.param(
        {
            "encode_state": True,
            "encode_time": False,
            "time_features_id": None,
            "num_time_features": num_time_features,
            "time_encoder_output_dim": time_encoder_output_dim,
        },
        id="enc-state",
    ),
    pytest.param(
        {
            "encode_state": True,
            "encode_time": False,
            "time_features_id": "ott-jax",
            "state_encoder_output_dim": state_encoder_output_dim,
        },
        id="enc-state-ottjax",
    ),
    pytest.param(
        {
            "encode_state": True,
            "encode_time": False,
            "time_features_id": "torch-cfm",
            "num_time_features": num_time_features,
            "time_encoder_output_dim": time_encoder_output_dim,
        },
        id="enc-state-cfm",
    ),
    pytest.param(
        {
            "encode_state": False,
            "encode_time": True,
            "time_features_id": None,
            "num_time_features": num_time_features,
            "state_encoder_output_dim": state_encoder_output_dim,
        },
        id="enc-time",
    ),
    pytest.param(
        {
            "encode_state": False,
            "encode_time": True,
            "time_features_id": "ott-jax",
            "time_encoder_output_dim": time_encoder_output_dim,
        },
        id="enc-time-ottjax",
    ),
    pytest.param(
        {
            "encode_state": False,
            "encode_time": True,
            "time_features_id": "torch-cfm",
            "num_time_features": num_time_features,
            "state_encoder_output_dim": state_encoder_output_dim,
            "time_encoder_output_dim": time_encoder_output_dim,
        },
        id="enc-time-cfm-dims",
    ),
    pytest.param(
        {
            "encode_state": False,
            "encode_time": False,
            "time_features_id": None,
            "state_encoder_output_dim": state_encoder_output_dim,
            "time_encoder_output_dim": time_encoder_output_dim,
        },
        id="enc-none",
    ),
    pytest.param(
        {
            "encode_state": False,
            "encode_time": False,
            "time_features_id": "ott-jax",
            "num_time_features": num_time_features,
        },
        id="enc-none-ottjax",
    ),
    pytest.param(
        {
            "encode_state": False,
            "encode_time": False,
            "time_features_id": "torch-cfm",
            "state_encoder_output_dim": state_encoder_output_dim,
            "time_encoder_output_dim": time_encoder_output_dim,
        },
        id="enc-none-cfm",
    ),
]

_BN = {"use_batchnorm": True, "batchnorm_affine": True, "batchnorm_track_running_stats": True}
_BN_NOAFFINE = {"use_batchnorm": True, "batchnorm_affine": False, "batchnorm_track_running_stats": True}
_BN_NOTRACK = {"use_batchnorm": True, "batchnorm_affine": True, "batchnorm_track_running_stats": False}
_BN_NOAFFINE_NOTRACK = {"use_batchnorm": True, "batchnorm_affine": False, "batchnorm_track_running_stats": False}

# The three sub-MLPs are constructed independently, so cross-multiplying their kwargs
# (5 * 5 * 5) tests nothing the diagonal plus a "only one of them" set does not.
MLP_KWARGS_CONFIGS = [
    pytest.param({}, id="mlpkwargs-none"),
    pytest.param(
        {
            "state_encoder_mlp_kwargs": _BN,
            "time_encoder_mlp_kwargs": _BN,
            "vf_decoder_mlp_kwargs": _BN,
        },
        id="mlpkwargs-all-bn",
    ),
    pytest.param(
        {
            "state_encoder_mlp_kwargs": _BN_NOAFFINE,
            "time_encoder_mlp_kwargs": _BN_NOAFFINE,
            "vf_decoder_mlp_kwargs": _BN_NOAFFINE,
        },
        id="mlpkwargs-all-bn-noaffine",
    ),
    pytest.param(
        {
            "state_encoder_mlp_kwargs": _BN_NOTRACK,
            "time_encoder_mlp_kwargs": _BN_NOTRACK,
            "vf_decoder_mlp_kwargs": _BN_NOTRACK,
        },
        id="mlpkwargs-all-bn-notrack",
    ),
    pytest.param(
        {
            "state_encoder_mlp_kwargs": _BN_NOAFFINE_NOTRACK,
            "time_encoder_mlp_kwargs": _BN_NOAFFINE_NOTRACK,
            "vf_decoder_mlp_kwargs": _BN_NOAFFINE_NOTRACK,
        },
        id="mlpkwargs-all-bn-noaffine-notrack",
    ),
    pytest.param({"state_encoder_mlp_kwargs": _BN}, id="mlpkwargs-state-bn"),
    pytest.param({"time_encoder_mlp_kwargs": _BN}, id="mlpkwargs-time-bn"),
    pytest.param({"vf_decoder_mlp_kwargs": _BN}, id="mlpkwargs-decoder-bn"),
]


def _assert_forward_shapes(vf: MLPVelocity) -> None:
    # case 0: x: (B, D) t: (B, 1)
    x = torch.zeros((batch_size, state_dim))
    t = torch.zeros((batch_size, 1))
    vt = vf(t, x)
    assert vt.shape == (batch_size, state_dim)

    # case 1: x: (B, D) t: (B, )
    x = torch.zeros((batch_size, state_dim))
    t = torch.zeros((batch_size,))
    vt = vf(t, x)
    assert vt.shape == (batch_size, state_dim)

    # case 2: x: (B, N, D) t: (B, 1)
    vf.eval()
    x = torch.zeros((batch_size, n_samples, state_dim))
    t = torch.zeros((batch_size, 1))
    vt = vf(t, x)
    assert vt.shape == (batch_size, n_samples, state_dim)

    # case 3: x: (B, N, D) t: (B, )
    x = torch.zeros((batch_size, n_samples, state_dim))
    t = torch.zeros((batch_size,))
    vt = vf(t, x)
    assert vt.shape == (batch_size, n_samples, state_dim)


class TestVF:
    @pytest.mark.parametrize("encoder_config", ENCODER_CONFIGS)
    @pytest.mark.parametrize("conditioning_id", [None, "concat", "resnet1d"])
    def test_vanilla_mlp_vf_forward(
        self,
        encoder_config: dict[str, Any],
        conditioning_id: str | None,
    ):
        vf = MLPVelocity(
            state_dim,
            conditioning_id=conditioning_id,
            **encoder_config,
        )
        _assert_forward_shapes(vf)

    @pytest.mark.parametrize("mlp_kwargs_config", MLP_KWARGS_CONFIGS)
    @pytest.mark.parametrize("conditioning_id", [None, "concat", "resnet1d"])
    def test_vanilla_mlp_vf_forward_batchnorm(
        self,
        mlp_kwargs_config: dict[str, Any],
        conditioning_id: str | None,
    ):
        vf = MLPVelocity(
            state_dim,
            encode_state=True,
            encode_time=True,
            time_features_id="torch-cfm",
            num_time_features=num_time_features,
            state_encoder_output_dim=state_encoder_output_dim,
            time_encoder_output_dim=time_encoder_output_dim,
            conditioning_id=conditioning_id,
            **mlp_kwargs_config,
        )
        _assert_forward_shapes(vf)
