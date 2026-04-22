from unittest.mock import Mock

import pytest
import torch

from sc_flow.backends.torch._types import PredictionData
from sc_flow.backends.torch.methods._utils import StepData
from sc_flow.backends.torch.methods.library._cfm import CFM
from sc_flow.data.containers._mixed_type import MixedTypeData


# -----------------------------------------------------------------------------
# Dummy module that replaces MLPVelocity for testing
# -----------------------------------------------------------------------------
class DummyModule(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)

    def forward(self, t, x, condition_dict=None, source=None):
        return self.linear(x)

    def get_vf_fn(self, **kwargs):
        return self.forward

    @classmethod
    def init_from_dims_registry(cls, dims_registry, *args, **kwargs):
        # Don't use dims_registry; just return an instance
        return cls(*args, **kwargs)


# -----------------------------------------------------------------------------
# Fixture
# -----------------------------------------------------------------------------
@pytest.fixture
def cfm_instance():
    dims_reg = Mock()
    dims_reg.feature_names = ["g1", "g2"]
    dims_reg.n_features = 2
    dm = Mock()

    original_module_cls = CFM._module_cls
    CFM._module_cls = DummyModule

    cfm = CFM(dims_registry=dims_reg, dm=dm, is_paired_setting=False, dtype=torch.float32, device_id="cpu")

    CFM._module_cls = original_module_cls
    cfm._module.forward = Mock(return_value=torch.randn(4, 2))
    return cfm


# Helper to create a mock condition/group data object that returns a BatchMixin-like object
def mock_condition_data():
    """Create a mock MixedTypeData whose extract_reps() returns an object with a 'mapping' attribute."""
    mock_data = Mock(spec=MixedTypeData)
    mock_reps = Mock()
    mock_reps.mapping = {}
    mock_data.extract_reps.return_value = mock_reps
    return mock_data


# -----------------------------------------------------------------------------
# Test suite for CFM
# -----------------------------------------------------------------------------
class TestCFM:
    """Tests for the Conditional Flow Matching (CFM) class."""

    def test_init_defaults(self, cfm_instance):
        assert cfm_instance._match_fn is not None
        assert cfm_instance._noise_sampler is not None
        assert cfm_instance._time_sampler is not None
        assert cfm_instance._probability_path is not None
        from sc_flow.backends.torch.coupling._coupling import independent_coupling

        assert cfm_instance._match_fn == independent_coupling

    def test_prepare_latent_state_generate_from_noise(self, cfm_instance):
        cfm_instance._generate_from_noise = True
        target = torch.randn(4, 2)
        latent = cfm_instance._prepare_latent_state(None, target)
        assert latent.shape == target.shape
        assert not torch.allclose(latent, target)

    def test_prepare_latent_state_use_source(self, cfm_instance):
        cfm_instance._generate_from_noise = False
        source = torch.randn(4, 2)
        target = torch.randn(4, 2)
        latent = cfm_instance._prepare_latent_state(source, target)
        assert latent is source

    def test_prepare_latent_state_source_none_fallback(self, cfm_instance):
        cfm_instance._generate_from_noise = False
        target = torch.randn(4, 2)
        latent = cfm_instance._prepare_latent_state(None, target)
        assert latent.shape == target.shape
        assert not torch.allclose(latent, target)

    def test_compute_loss(self, cfm_instance):
        mock_cond = mock_condition_data()
        mock_group = mock_condition_data()

        step_data = StepData(
            target_state=torch.randn(4, 2),
            target_coupling_lin=None,
            target_coupling_quad=None,
            target_condition_data=mock_cond,
            target_group_data=mock_group,
            source_state=torch.randn(4, 2),
            source_coupling_lin=None,
            source_coupling_quad=None,
            source_condition_data=mock_cond,
            source_group_data=mock_group,
        )
        cfm_instance._probability_path = Mock()
        cfm_instance._probability_path.compute_xt.return_value = torch.randn(4, 2)
        cfm_instance._probability_path.compute_ut.return_value = torch.randn(4, 2)
        cfm_instance._module = Mock(return_value=torch.randn(4, 2))
        loss, info = cfm_instance._compute_loss(step_data)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert "loss" in info
        assert isinstance(info["loss"], float)

    def test_predict_no_trajectory(self, cfm_instance):
        step_data = Mock()
        step_data.target_state = torch.randn(4, 2)
        step_data.source_state = torch.randn(4, 2)
        step_data.target_condition_data = mock_condition_data()
        step_data.target_group_data = mock_condition_data()
        mock_solver = Mock()
        mock_solver.solve.return_value = torch.randn(4, 2)

        # Replace the default solver class with a mock that returns our mock solver
        cfm_instance._default_solver_cls = Mock(return_value=mock_solver)

        pred = cfm_instance._predict(step_data, return_trajectory=False, num_steps=10)

        assert isinstance(pred, PredictionData)
        assert pred.samples is not None
        assert pred.traj is None
        mock_solver.solve.assert_called_once()
        _, kwargs = mock_solver.solve.call_args
        assert kwargs.get("return_trajectory") is False

    def test_predict_with_trajectory(self, cfm_instance):
        step_data = Mock()
        step_data.target_state = torch.randn(4, 2)
        step_data.source_state = torch.randn(4, 2)
        step_data.target_condition_data = mock_condition_data()
        step_data.target_group_data = mock_condition_data()
        mock_solver = Mock()
        # Simulate a trajectory tensor: (num_steps, batch, dim)
        traj_tensor = torch.stack([torch.randn(4, 2) for _ in range(10)])
        mock_solver.solve.return_value = traj_tensor

        cfm_instance._default_solver_cls = Mock(return_value=mock_solver)

        pred = cfm_instance._predict(step_data, return_trajectory=True, num_steps=10)

        assert pred.traj is traj_tensor
        assert torch.equal(pred.samples, traj_tensor[-1])

    def test_predict_custom_solver(self, cfm_instance):
        step_data = Mock()
        step_data.target_state = torch.randn(4, 2)
        step_data.source_state = torch.randn(4, 2)
        step_data.target_condition_data = mock_condition_data()
        step_data.target_group_data = mock_condition_data()
        mock_solver_cls = Mock()
        mock_solver = Mock()
        mock_solver.solve.return_value = torch.randn(4, 2)
        mock_solver_cls.return_value = mock_solver
        cfm_instance._predict(step_data, solver_cls=mock_solver_cls, num_steps=5)
        mock_solver_cls.assert_called_once()
        call_kwargs = mock_solver_cls.call_args[1]
        assert call_kwargs.get("method") == "euler"

    def test_train_step(self, cfm_instance):
        matched_distr = Mock()
        step_data = Mock()
        cfm_instance._extract_step_data = Mock(return_value=step_data)
        cfm_instance._train_step_forward = Mock(return_value=(torch.tensor(0.5), {"loss": 0.5}))
        loss, log_dict = cfm_instance.train_step(matched_distr)
        assert loss == torch.tensor(0.5)
        assert log_dict == {"loss": 0.5}

    def test_train_step_forward_integration(self, cfm_instance):
        step_data = Mock()
        cfm_instance._extract_matched_observations = Mock(return_value=step_data)
        cfm_instance._compute_loss = Mock(return_value=(torch.tensor(0.3), {"loss": 0.3}))
        loss, info = cfm_instance._train_step_forward(step_data)
        assert loss == 0.3
        cfm_instance._extract_matched_observations.assert_called_once_with(step_data)
