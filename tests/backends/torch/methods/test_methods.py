from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from sc_flow.backends.torch._types import PredictionData
from sc_flow.backends.torch.methods._base import TorchBaseMethod, TorchGenerativeFlow
from sc_flow.backends.torch.methods._utils import StepData
from sc_flow.backends.torch.nn._modules import BaseModule
from sc_flow.data._composite import MatchedDistributions
from sc_flow.data.containers._coupling import CouplingData
from sc_flow.data.containers._distribution import DistributionData
from sc_flow.data.containers._state import StateData


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

    def train_step(self, *args, **kwargs):
        pass

    def predict(self, *args, **kwargs):
        pass

    def _compute_loss(self, *args, **kwargs):
        pass

    def _predict(self, *args, **kwargs):
        pass


class DummyTorchGenerativeFlow(TorchGenerativeFlow):
    _module_cls = DummyModule
    _default_solver_cls = Mock()

    def _compute_loss(self, step_data, *args, **kwargs):
        return torch.tensor(0.0), {}

    def _predict(self, step_data, *args, **kwargs):
        return PredictionData(samples=torch.randn(1, 2), traj=None)


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
    """Create a DummyTorchMethod with optimization manager patched out."""
    with patch("sc_flow.backends.torch.methods._base.OptimizationManager.init_from_module") as mock_init:
        mock_init.return_value = Mock()
        method = DummyTorchMethod(mock_dims_registry, mock_data_manager, False, dtype=torch.float32, device_id="cpu")
    return method


@pytest.fixture
def torch_gen_flow(mock_dims_registry, mock_data_manager):
    with patch("sc_flow.backends.torch.methods._base.OptimizationManager.init_from_module") as mock_init:
        mock_init.return_value = Mock()
        flow = DummyTorchGenerativeFlow(
            mock_dims_registry, mock_data_manager, False, dtype=torch.float32, device_id="cpu"
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

    def test_batchmixin_to_torch(self, torch_method):
        batch_mixin = Mock()
        batch_mixin.mapping = {"a": np.array([1, 2]), "b": np.array([3.0, 4.0])}
        result = torch_method._batchmixin_to_torch(batch_mixin)
        assert isinstance(result["a"], torch.Tensor)
        assert result["a"].dtype == torch.float32
        assert result["a"].tolist() == [1, 2]
        assert result["b"].tolist() == [3.0, 4.0]

    def test_extract_state_data(self, torch_method):
        state = Mock()
        state.X = np.random.randn(5, 2)
        tensor = torch_method._extract_state_data(state)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (5, 2)
        assert torch_method._extract_state_data(None) is None

    def test_extract_coupling_data(self, torch_method):
        dist_data = Mock(spec=DistributionData)
        coupling = Mock(spec=CouplingData)
        coupling.state_lin = Mock()
        coupling.state_lin.X = np.random.randn(3, 2)
        coupling.state_quad = None  # quadratic term is None
        dist_data.source_coupling_data = coupling

        # Mock _extract_state_data to return tensor for state_lin, None for state_quad
        with patch.object(torch_method, "_extract_state_data") as mock_extract:
            # First call (for state_lin) returns tensor, second call (state_quad) returns None
            mock_extract.side_effect = [torch.randn(3, 2), None]
            lin, quad = torch_method._extract_coupling_data(dist_data, "source")
            assert lin is not None
            assert quad is None

    def test_extract_distribution_data(self, torch_method):
        # Create real StateData with a numpy array
        state_data = StateData(X=np.random.randn(5, 2))
        dist_data = Mock(spec=DistributionData)
        dist_data.state_data = state_data
        dist_data.condition_data = Mock()
        dist_data.groups_data = Mock()
        coupling = Mock()
        coupling.state_lin = Mock()
        coupling.state_quad = None
        dist_data.source_coupling_data = coupling

        with patch.object(torch_method, "_extract_coupling_data", return_value=(Mock(), None)):
            result = torch_method._extract_distribution_data(dist_data, "source")
            assert len(result) == 5

    def test_extract_step_data(self, torch_method):
        matched = Mock(spec=MatchedDistributions)
        source_dist = Mock(spec=DistributionData)
        target_dist = Mock(spec=DistributionData)
        matched.source_distribution = source_dist
        matched.target_distribution = target_dist
        with patch.object(torch_method, "_extract_distribution_data", return_value=(None, None, None, None, None)):
            step_data = torch_method._extract_step_data(matched)
            assert isinstance(step_data, StepData)

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

    def test_extract_matched_observations_no_match(self, torch_gen_flow):
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
        new_step = torch_gen_flow._extract_matched_observations(step_data)
        assert new_step.target_state is step_data.target_state
        assert new_step.source_state is step_data.source_state

    def test_extract_matched_observations_with_match(self, torch_gen_flow):
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

        new_step = torch_gen_flow._extract_matched_observations(step_data)
        assert new_step.target_state.shape[0] == 2
        assert new_step.source_state.shape[0] == 2
        assert new_step.target_condition_data.shape[0] == 2
        assert new_step.source_condition_data.shape[0] == 2

    def test_train_step_forward(self, torch_gen_flow):
        step_data = Mock()
        torch_gen_flow._compute_loss = Mock(return_value=(torch.tensor(0.5), {"loss": 0.5}))
        loss, info = torch_gen_flow._train_step_forward(step_data)
        assert loss == 0.5
        assert info["loss"] == 0.5

    def test_train_step(self, torch_gen_flow):
        matched_distr = Mock()
        step_data = Mock()
        torch_gen_flow._extract_step_data = Mock(return_value=step_data)
        torch_gen_flow._train_step_forward = Mock(return_value=(torch.tensor(1.0), {"loss": 1.0}))
        torch_gen_flow._optimization_manager = Mock()
        torch_gen_flow._optimization_manager.backward_pass = Mock()
        result = torch_gen_flow.train_step(matched_distr)
        assert result == {"loss": 1.0}
        torch_gen_flow._optimization_manager.backward_pass.assert_called_once_with(torch.tensor(1.0))

    def test_predict_with_no_grad(self, torch_gen_flow):
        matched_distr = Mock()
        step_data = Mock()
        torch_gen_flow._extract_step_data = Mock(return_value=step_data)
        pred_data = PredictionData(samples=torch.randn(4, 2), traj=None)
        torch_gen_flow._predict = Mock(return_value=pred_data)
        with patch("torch.no_grad"):
            result = torch_gen_flow.predict(matched_distr, no_grad=True)
            assert result is pred_data
