# tests/backends/torch/nn/test_fm.py
from unittest.mock import Mock

import pytest
import torch

from sc_flow.backends.torch.nn._fm import MLPFlowMap
from sc_flow.data._dims_registry import DataDimensionalitiesRegistry


class TestMLPFlowMap:
    """Tests for MLPFlowMap neural network module."""

    @pytest.fixture
    def dims_registry(self):
        registry = Mock(spec=DataDimensionalitiesRegistry)
        registry.state_dim = 10
        registry.condition_reps_dims = {}
        registry.condition_continuous_dims = {}
        registry.groups_reps_dims = {}
        return registry

    def test_init_from_dims_registry(self, dims_registry):
        module = MLPFlowMap.init_from_dims_registry(dims_registry)
        assert isinstance(module, MLPFlowMap)
        assert module._state_dim == 10

    def test_forward_shape(self):
        module = MLPFlowMap(state_dim=2, encode_state=True, encode_time=True)
        batch_size = 4
        s = torch.rand(batch_size)
        t = torch.rand(batch_size)
        x = torch.randn(batch_size, 2)
        out = module(s, t, x)
        assert out.shape == (batch_size, 2)

    def test_forward_with_condition(self):
        module = MLPFlowMap(state_dim=2, encode_state=True, encode_time=True)
        batch_size = 4
        s = torch.rand(batch_size)
        t = torch.rand(batch_size)
        x = torch.randn(batch_size, 2)
        condition_dict = {"cond1": torch.randn(batch_size, 3)}
        out = module(s, t, x, condition_dict=condition_dict)
        assert out.shape == (batch_size, 2)

    def test_forward_with_source(self):
        module = MLPFlowMap(
            state_dim=2,
            encode_state=True,
            encode_time=True,
            source_encoder_mlp_kwargs={"input_dim": 2, "hidden_dims": [4], "output_dim": 4},
        )
        batch_size = 4
        s = torch.rand(batch_size)
        t = torch.rand(batch_size)
        x = torch.randn(batch_size, 2)
        source = torch.randn(batch_size, 2)
        out = module(s, t, x, source=source)
        assert out.shape == (batch_size, 2)

    def test_get_vf_fn(self):
        module = MLPFlowMap(state_dim=2)
        vf_fn = module.get_vf_fn(condition_dict={}, source=None)
        assert callable(vf_fn)
