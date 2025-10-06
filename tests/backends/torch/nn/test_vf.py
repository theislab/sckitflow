from collections.abc import Callable
from typing import Any

import pytest
import torch

from sc_flow._types import LayersDict, NestedLayersDict, TimeFeaturesId
from sc_flow.backends.torch._types import MappedTensor, TTimeFeaturesFn
from sc_flow.backends.torch.nn._vf import (
    MLPUnconditionalVF,
)

batch_size = 32
n_samples = 50
state_dim = 20
num_time_features = 256
state_encoder_output_dim = 8
time_encoder_output_dim = 4
n_combs = 3
condition_encoder_output_dim = 6
condition0_input_dim = 4
condition0_output_dim = 2
condition1_input_dim = 8
condition1_output_dim = 4

# dictionaries
input_layers_double_condition = {
    "condition0": {
        "input_dim": condition0_input_dim,
        "output_dim": condition0_output_dim,
    },
    "condition1": {
        "input_dim": condition1_input_dim,
        "output_dim": condition1_output_dim,
    },
}
condition_dict = {
    covariate: torch.zeros((batch_size, n_combs, layers["input_dim"]))
    for covariate, layers in input_layers_double_condition.items()
}


class TestVF:
    @pytest.mark.parametrize("condition_dict", [None, condition_dict])
    @pytest.mark.parametrize("encode_state", [True, False])
    @pytest.mark.parametrize("encode_time", [True, False])
    @pytest.mark.parametrize("time_features_id", ["ott-jax", "torch-cfm", None])
    @pytest.mark.parametrize("time_features_fn", [None])
    @pytest.mark.parametrize("num_time_features", [None, num_time_features])
    @pytest.mark.parametrize("time_features_kwargs", [None])
    @pytest.mark.parametrize("state_encoder_output_dim", [None, state_encoder_output_dim])
    @pytest.mark.parametrize("time_encoder_output_dim", [None, time_encoder_output_dim])
    @pytest.mark.parametrize(
        "state_encoder_mlp_kwargs",
        [
            None,
            {
                "use_batchnorm": True,
                "batchnorm_affine": True,
                "batchnorm_track_running_stats": True,
            },
            {
                "use_batchnorm": True,
                "batchnorm_affine": False,
                "batchnorm_track_running_stats": True,
            },
            {
                "use_batchnorm": True,
                "batchnorm_affine": True,
                "batchnorm_track_running_stats": False,
            },
            {
                "use_batchnorm": True,
                "batchnorm_affine": False,
                "batchnorm_track_running_stats": False,
            },
        ],
    )
    @pytest.mark.parametrize(
        "time_encoder_mlp_kwargs",
        [
            None,
            {
                "use_batchnorm": True,
                "batchnorm_affine": True,
                "batchnorm_track_running_stats": True,
            },
            {
                "use_batchnorm": True,
                "batchnorm_affine": False,
                "batchnorm_track_running_stats": True,
            },
            {
                "use_batchnorm": True,
                "batchnorm_affine": True,
                "batchnorm_track_running_stats": False,
            },
            {
                "use_batchnorm": True,
                "batchnorm_affine": False,
                "batchnorm_track_running_stats": False,
            },
        ],
    )
    @pytest.mark.parametrize(
        "vf_decoder_mlp_kwargs",
        [
            None,
            {
                "use_batchnorm": True,
                "batchnorm_affine": True,
                "batchnorm_track_running_stats": True,
            },
            {
                "use_batchnorm": True,
                "batchnorm_affine": False,
                "batchnorm_track_running_stats": True,
            },
            {
                "use_batchnorm": True,
                "batchnorm_affine": True,
                "batchnorm_track_running_stats": False,
            },
            {
                "use_batchnorm": True,
                "batchnorm_affine": False,
                "batchnorm_track_running_stats": False,
            },
        ],
    )
    @pytest.mark.parametrize("conditioning_id", [None, "concat", "resnet1d"])
    @pytest.mark.parametrize("conditioning_fn", [None])
    @pytest.mark.parametrize("conditioning_kwargs", [None])
    @pytest.mark.parametrize("condition_encoder_input_layers", [None, input_layers_double_condition])
    @pytest.mark.parametrize("condition_encoder_output_dim", [None, condition_encoder_output_dim])
    @pytest.mark.parametrize("condition_encoder_pooling_mode", ["sum", "mean"])
    @pytest.mark.parametrize("condition_encoder_pooling_kwargs", [None, {}])
    @pytest.mark.parametrize("condition_encoder_output_layers_kwargs", [None, {}])
    def test_vanilla_mlp_vf_forward(
        self,
        condition_dict: MappedTensor | None,
        encode_state: bool,
        encode_time: bool,
        time_features_id: TimeFeaturesId | None,
        time_features_fn: TTimeFeaturesFn | None,
        num_time_features: int | None,
        time_features_kwargs: dict[str, Any] | None,
        state_encoder_output_dim: int | None,
        time_encoder_output_dim: int | None,
        state_encoder_mlp_kwargs: dict[str, Any] | None,
        time_encoder_mlp_kwargs: dict[str, Any] | None,
        vf_decoder_mlp_kwargs: dict[str, Any] | None,
        conditioning_id: str | None,
        conditioning_fn: Callable | None,
        conditioning_kwargs: dict[str, Any],
        condition_encoder_input_layers: None | NestedLayersDict,
        condition_encoder_output_dim: int,
        condition_encoder_pooling_mode: str,
        condition_encoder_pooling_kwargs: dict[str, Any],
        condition_encoder_output_layers_kwargs: None | LayersDict,
    ):
        vf = MLPUnconditionalVF(
            state_dim,
            encode_state=encode_state,
            encode_time=encode_time,
            time_features_id=time_features_id,
            time_features_fn=time_features_fn,
            num_time_features=num_time_features,
            time_features_kwargs=time_features_kwargs,
            state_encoder_output_dim=state_encoder_output_dim,
            time_encoder_output_dim=time_encoder_output_dim,
            state_encoder_mlp_kwargs=state_encoder_mlp_kwargs,
            time_encoder_mlp_kwargs=time_encoder_mlp_kwargs,
            vf_decoder_mlp_kwargs=vf_decoder_mlp_kwargs,
            conditioning_id=conditioning_id,
            conditioning_fn=conditioning_fn,
            conditioning_kwargs=conditioning_kwargs,
            condition_encoder_input_layers=condition_encoder_input_layers,
            condition_encoder_output_dim=condition_encoder_output_dim,
            condition_encoder_pooling_mode=condition_encoder_pooling_mode,
            condition_encoder_pooling_kwargs=condition_encoder_pooling_kwargs,
            condition_encoder_output_layers_kwargs=condition_encoder_output_layers_kwargs,
        )

        # case 0: x: (B, D) t: (B, 1)
        x = torch.zeros((batch_size, state_dim))
        t = torch.zeros((batch_size, 1))
        if vf.is_conditional and condition_dict is None:
            with pytest.raises(TypeError):
                vt = vf(t, x, condition_dict=condition_dict)
        else:
            vt = vf(t, x, condition_dict=condition_dict)
        assert vt.shape == (batch_size, state_dim)

        # case 1: x: (B, D) t: (B, )
        x = torch.zeros((batch_size, state_dim))
        t = torch.zeros((batch_size,))
        if vf.is_conditional and condition_dict is None:
            with pytest.raises(TypeError):
                vt = vf(t, x, condition_dict=condition_dict)
        else:
            vt = vf(t, x, condition_dict=condition_dict)
        assert vt.shape == (batch_size, state_dim)

        # case 2: x: (B, N, D) t: (B, 1)
        vf.eval()
        x = torch.zeros((batch_size, n_samples, state_dim))
        t = torch.zeros((batch_size, 1))
        if vf.is_conditional and condition_dict is None:
            with pytest.raises(TypeError):
                vt = vf(t, x, condition_dict=condition_dict)
        else:
            vt = vf(t, x, condition_dict=condition_dict)
        assert vt.shape == (batch_size, n_samples, state_dim)

        # case 2: x: (B, N, D) t: (B, )
        x = torch.zeros((batch_size, n_samples, state_dim))
        t = torch.zeros((batch_size,))
        if vf.is_conditional and condition_dict is None:
            with pytest.raises(TypeError):
                vt = vf(t, x, condition_dict=condition_dict)
        else:
            vt = vf(t, x, condition_dict=condition_dict)
        assert vt.shape == (batch_size, n_samples, state_dim)
