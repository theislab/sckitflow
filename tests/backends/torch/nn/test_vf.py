from collections.abc import Callable
from typing import Any

import pytest
import torch

from sc_flow._types import LayersDict, NestedLayersDict, TimeFeaturesId
from sc_flow.backends.torch._types import MappedTensor, TTimeFeaturesFn
from sc_flow.backends.torch.nn._vf import (
    BaseVelocityField,
    MLPUnconditionalVF,
)

batch_size = 32
n_samples = 50
state_dim = 20
source_input_dim = 15
source_encoder_output_dim = 4
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
    @staticmethod
    def verify_output(
        source_encoder_mlp_kwargs: LayersDict,
        vf: BaseVelocityField,
        t: torch.Tensor,
        x: torch.Tensor,
        condition_dict: MappedTensor | None = None,
        source: torch.Tensor | None = None,
    ) -> None:
        if vf.is_conditional and condition_dict is None:
            with pytest.raises(TypeError):
                vt = vf(t, x, condition_dict=condition_dict, source=source)
            return None
        elif vf.use_source_encoder and source is None:
            with pytest.raises(TypeError):
                vt = vf(t, x, condition_dict=condition_dict, source=source)
            return None
        elif vf.use_source_encoder and "input_dim" not in source_encoder_mlp_kwargs:
            if source.shape[-1] != x.shape[-1]:
                with pytest.raises(RuntimeError):
                    vt = vf(t, x, condition_dict=condition_dict, source=source)
                return None
        elif vf.use_source_encoder and "input_dim" in source_encoder_mlp_kwargs:
            if source.shape[-1] != source_encoder_mlp_kwargs["input_dim"]:
                with pytest.raises(RuntimeError):
                    vt = vf(t, x, condition_dict=condition_dict, source=source)
                return None
        vt = vf(t, x, condition_dict=condition_dict, source=source)
        assert vt.shape == x.shape

    @pytest.mark.parametrize("condition_dict", [None, condition_dict])
    @pytest.mark.parametrize("encode_state", [True, False])
    @pytest.mark.parametrize("encode_time", [True, False])
    @pytest.mark.parametrize("time_features_id", ["ott-jax", "torch-cfm", None])
    @pytest.mark.parametrize("time_features_fn", [None])
    @pytest.mark.parametrize("num_time_features", [None, num_time_features])
    @pytest.mark.parametrize("time_features_kwargs", [None])
    @pytest.mark.parametrize("state_encoder_output_dim", [None, state_encoder_output_dim])
    @pytest.mark.parametrize("time_encoder_output_dim", [None, time_encoder_output_dim])
    @pytest.mark.parametrize("conditioning_id", [None, "concat", "resnet1d"])
    @pytest.mark.parametrize("conditioning_fn", [None])
    @pytest.mark.parametrize("conditioning_kwargs", [None])
    @pytest.mark.parametrize("condition_encoder_input_layers", [None, input_layers_double_condition])
    @pytest.mark.parametrize("condition_encoder_output_dim", [None, condition_encoder_output_dim])
    @pytest.mark.parametrize("condition_encoder_pooling_mode", ["sum", "mean"])
    @pytest.mark.parametrize("condition_encoder_pooling_kwargs", [None, {}])
    @pytest.mark.parametrize("condition_encoder_output_layers_kwargs", [None, {}])
    @pytest.mark.parametrize("source_encoder_mlp_kwargs", [{}, {"input_dim": source_input_dim}])
    @pytest.mark.parametrize("source_encoder_output_dim", [None, source_encoder_output_dim])
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
        conditioning_id: str | None,
        conditioning_fn: Callable | None,
        conditioning_kwargs: dict[str, Any],
        condition_encoder_input_layers: None | NestedLayersDict,
        condition_encoder_output_dim: int,
        condition_encoder_pooling_mode: str,
        condition_encoder_pooling_kwargs: dict[str, Any],
        condition_encoder_output_layers_kwargs: None | LayersDict,
        source_encoder_mlp_kwargs: None | LayersDict,
        source_encoder_output_dim: int | None,
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
            conditioning_id=conditioning_id,
            conditioning_fn=conditioning_fn,
            conditioning_kwargs=conditioning_kwargs,
            condition_encoder_input_layers=condition_encoder_input_layers,
            condition_encoder_output_dim=condition_encoder_output_dim,
            condition_encoder_pooling_mode=condition_encoder_pooling_mode,
            condition_encoder_pooling_kwargs=condition_encoder_pooling_kwargs,
            condition_encoder_output_layers_kwargs=condition_encoder_output_layers_kwargs,
            source_encoder_mlp_kwargs=source_encoder_mlp_kwargs,
            source_encoder_output_dim=source_encoder_output_dim,
        )

        # case 0: x: (B, D) t: (B, 1)
        x = torch.zeros((batch_size, state_dim))
        t = torch.zeros((batch_size, 1))
        source_cases = [None, torch.zeros_like(x), torch.zeros(batch_size, source_input_dim)]
        for source_state in source_cases:
            self.verify_output(source_encoder_mlp_kwargs, vf, t, x, condition_dict, source_state)

        # case 1: x: (B, D) t: (B, )
        x = torch.zeros((batch_size, state_dim))
        t = torch.zeros((batch_size,))
        source_cases = [None, torch.zeros_like(x), torch.zeros(batch_size, source_input_dim)]
        for source_state in source_cases:
            self.verify_output(source_encoder_mlp_kwargs, vf, t, x, condition_dict, source_state)

        # case 2: x: (B, N, D) t: (B, 1)
        vf.eval()
        x = torch.zeros((batch_size, n_samples, state_dim))
        t = torch.zeros((batch_size, 1))
        source_cases = [None, torch.zeros_like(x), torch.zeros(batch_size, n_samples, source_input_dim)]
        for source_state in source_cases:
            self.verify_output(source_encoder_mlp_kwargs, vf, t, x, condition_dict, source_state)

        # case 2: x: (B, N, D) t: (B, )
        x = torch.zeros((batch_size, n_samples, state_dim))
        t = torch.zeros((batch_size,))
        source_cases = [None, torch.zeros_like(x), torch.zeros(batch_size, n_samples, source_input_dim)]
        for source_state in source_cases:
            self.verify_output(source_encoder_mlp_kwargs, vf, t, x, condition_dict, source_state)
