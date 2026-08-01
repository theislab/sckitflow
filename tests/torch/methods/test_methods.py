from unittest.mock import Mock

import pytest
import torch

from sckitflow.core._types import PredictionData, StepData
from sckitflow.core.methods._base import TorchBaseMethod, TorchGenerativeFlow
from sckitflow.core.nn._modules import BaseModule


# -----------------------------------------------------------------------------
# Minimal torch module that can be instantiated
# -----------------------------------------------------------------------------
class DummyModule(BaseModule):
    """A minimal module with parameters to satisfy optimizers."""

    def __init__(self):
        super().__init__()

    def _make_modules(self):
        self.linear = torch.nn.Linear(2, 2)

    def forward(self, t, x, condition_dict=None, source=None):
        return self.linear(x)


# -----------------------------------------------------------------------------
# Concrete test subclasses that use DummyModule
# -----------------------------------------------------------------------------
class DummyTorchMethod(TorchBaseMethod):
    _module_cls = DummyModule

    def _step_fn(self, *args, **kwargs):
        return torch.tensor(0.0), {}

    def _predict(self, *args, **kwargs):
        pass

    def _compute_loss(self, *args, **kwargs):
        pass

    def _predict(self, *args, **kwargs):
        pass


class DummyTorchGenerativeFlow(TorchGenerativeFlow):
    _module_cls = DummyModule
    _default_solver_cls = Mock()

    def _step_fn(self, step_data, *args, **kwargs):
        return torch.tensor(0.0), {}

    def _predict(self, step_data, *args, **kwargs):
        return PredictionData(X=torch.randn(1, 2), traj=None)

    def train_step(self, *args, **kwargs):
        return torch.tensor(0.0), {}


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture
def mock_dims_registry():
    reg = Mock()
    reg.feature_names = ["g1", "g2"]
    reg.n_features = 2
    return reg


@pytest.fixture
def mock_data_manager():
    return Mock()


@pytest.fixture
def torch_method(mock_dims_registry, mock_data_manager):
    method = DummyTorchMethod(
        dims_registry=mock_dims_registry,
        dm=mock_data_manager,
        dtype=torch.float32,
        device_id="cpu",
    )
    return method


@pytest.fixture
def torch_gen_flow(mock_dims_registry, mock_data_manager):
    flow = DummyTorchGenerativeFlow(
        dims_registry=mock_dims_registry,
        dm=mock_data_manager,
        dtype=torch.float32,
        device_id="cpu",
    )
    flow._match_fn = Mock(return_value=(None, None))
    return flow


# -----------------------------------------------------------------------------
# Test suite for TorchBaseMethod
# -----------------------------------------------------------------------------
class TestTorchBaseMethod:
    """Tests for TorchBaseMethod utility methods."""

    def test_safe_subscript_obj(self, torch_method):
        data = torch.tensor([1, 2, 3, 4])
        idx = torch.tensor([0, 2])
        result = torch_method._safe_subscript_obj(data, idx)
        assert torch.equal(result, torch.tensor([1, 3]))
        assert torch_method._safe_subscript_obj(None, idx) is None
        assert torch_method._safe_subscript_obj(data, None) is data

    def test_set_train_mode(self, torch_method):
        torch_method.module.train = Mock()
        torch_method.module.eval = Mock()
        torch_method.set_train_mode(True)
        torch_method.module.train.assert_called_once()
        torch_method.set_train_mode(False)
        torch_method.module.eval.assert_called_once()


# -----------------------------------------------------------------------------
# Test suite for TorchGenerativeFlow
# -----------------------------------------------------------------------------
class TestTorchGenerativeFlow:
    """Tests for TorchGenerativeFlow matching and training logic."""

    def test_call_match_fn_safe_no_source(self, torch_gen_flow):
        src_lin = src_quad = None
        tgt_lin = Mock()
        tgt_quad = Mock()
        src_idx, tgt_idx = torch_gen_flow._call_match_fn_safe(src_lin, src_quad, tgt_lin, tgt_quad)
        assert src_idx is None
        assert tgt_idx is None
        torch_gen_flow._match_fn.assert_not_called()

    def test_call_match_fn_safe_with_source(self, torch_gen_flow):
        src_lin = torch.randn(4, 2)
        src_quad = None
        tgt_lin = torch.randn(4, 2)
        tgt_quad = None
        torch_gen_flow._match_fn.return_value = (torch.tensor([0, 1]), torch.tensor([2, 3]))
        src_idx, tgt_idx = torch_gen_flow._call_match_fn_safe(src_lin, src_quad, tgt_lin, tgt_quad)
        torch_gen_flow._match_fn.assert_called_once_with(
            source_lin=src_lin, target_lin=tgt_lin, source_quad=src_quad, target_quad=tgt_quad
        )
        assert torch.equal(src_idx, torch.tensor([0, 1]))
        assert torch.equal(tgt_idx, torch.tensor([2, 3]))

    def test_match_observations_no_match(self, torch_gen_flow):
        step_data = StepData(
            target_state=torch.randn(4, 2),
            target_coupling_lin=None,
            target_coupling_quad=None,
            target_condition_data={"a": torch.randn(4, 3)},
            target_group_data=None,
            source_state=torch.randn(4, 2),
            source_coupling_lin=None,
            source_coupling_quad=None,
            source_condition_data={"a": torch.randn(4, 3)},
            source_group_data=None,
        )
        torch_gen_flow._match_fn = Mock(return_value=(None, None))
        new_step = torch_gen_flow._match_observations(step_data)
        assert new_step.target_state is step_data.target_state
        assert new_step.source_state is step_data.source_state

    def test_match_observations_with_match(self, torch_gen_flow):
        # Use tensors for condition_data (not dict) to allow indexing
        step_data = StepData(
            target_state=torch.randn(4, 2),
            target_coupling_lin=torch.randn(4, 2),
            target_coupling_quad=None,
            target_condition_data=torch.randn(4, 3),  # tensor
            target_group_data=None,
            source_state=torch.randn(4, 2),
            source_coupling_lin=torch.randn(4, 2),
            source_coupling_quad=None,
            source_condition_data=torch.randn(4, 3),  # tensor
            source_group_data=None,
        )
        src_idx = torch.tensor([0, 2])
        tgt_idx = torch.tensor([1, 3])
        torch_gen_flow._match_fn = Mock(return_value=(src_idx, tgt_idx))

        new_step = torch_gen_flow._match_observations(step_data)
        assert new_step.target_state.shape[0] == 2
        assert new_step.source_state.shape[0] == 2
        assert new_step.target_condition_data.shape[0] == 2
        assert new_step.source_condition_data.shape[0] == 2

    def test_train_step_forward(self, torch_gen_flow):
        step_data = Mock()
        torch_gen_flow._step_fn = Mock(return_value=(torch.tensor(0.5), {"loss": 0.5}))
        loss, info = torch_gen_flow._train_step_forward(step_data)
        assert loss == 0.5
        assert info["loss"] == 0.5

    def test_predict_with_no_grad(self, torch_gen_flow):
        step_data = StepData(
            target_state=torch.randn(4, 2),
            target_coupling_lin=None,
            target_coupling_quad=None,
            target_condition_data=None,
            target_group_data=None,
            source_state=None,
            source_coupling_lin=None,
            source_coupling_quad=None,
            source_condition_data=None,
            source_group_data=None,
        )
        pred_data = PredictionData(X=torch.randn(4, 2), traj=None)
        torch_gen_flow._predict = Mock(return_value=pred_data)
        result = torch_gen_flow.predict(step_data, no_grad=True)
        assert result is pred_data
        torch_gen_flow._predict.assert_called_once_with(step_data)
