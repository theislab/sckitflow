from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from sckitflow.core._types import PredictionData, StepData
from sckitflow.core.methods.library._cfm import CFM


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


class DummyMixedTypeData:
    def __getitem__(self, idx):
        return self

    def extract_reps(self):
        return Mock(mapping={})


def mock_condition_data():
    return {"a": torch.randn(4, 3)}


# -----------------------------------------------------------------------------
# Test suite for CFM
# -----------------------------------------------------------------------------
class TestCFM:
    def test_init_defaults(self, cfm_instance):
        assert cfm_instance._match_fn is not None
        assert cfm_instance._noise_sampler is not None
        assert cfm_instance._time_sampler is not None
        assert cfm_instance._probability_path is not None
        from sckitflow.core.coupling._coupling import independent_coupling

        assert cfm_instance._match_fn == independent_coupling

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
        # `_module` is a registered child module and cannot be swapped for a plain Mock;
        # the fixture already stubs its `forward`.
        loss, info = cfm_instance.compute_loss(step_data)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert "loss" in info
        assert isinstance(info["loss"], float)

    # -------------------------------------------------------------------------
    # Prediction tests
    # -------------------------------------------------------------------------
    def test_predict_no_trajectory(self, cfm_instance):
        step_data = Mock()
        step_data.target_state = torch.randn(4, 2)
        step_data.source_state = torch.randn(4, 2)
        step_data.target_condition_data = mock_condition_data()
        step_data.target_group_data = mock_condition_data()
        mock_solver = Mock()
        # Final state only: shape (batch, dim) -> (4, 2)
        mock_solver.solve.return_value = torch.randn(4, 2)

        cfm_instance._default_solver_cls = Mock(return_value=mock_solver)

        pred = cfm_instance.infer(step_data, return_trajectory=False, num_steps=10)

        assert isinstance(pred, PredictionData)
        # X should be 2D (batch, dim)
        assert pred.X.shape == (4, 2)
        # raw_samples is None because n_samples is None (single sample)
        assert pred.raw_samples is None
        assert pred.traj is None
        mock_solver.solve.assert_called_once()
        _, kwargs = mock_solver.solve.call_args
        assert kwargs.get("return_trajectory") is False

    def test_predict_with_trajectory_single_sample(self, cfm_instance):
        step_data = Mock()
        step_data.target_state = torch.randn(4, 2)
        step_data.source_state = torch.randn(4, 2)
        step_data.target_condition_data = mock_condition_data()
        step_data.target_group_data = mock_condition_data()
        mock_solver = Mock()
        # Simulate trajectory: (num_steps, batch, dim)
        traj_tensor = torch.stack([torch.randn(4, 2) for _ in range(10)])  # (10,4,2)
        mock_solver.solve.return_value = traj_tensor

        cfm_instance._default_solver_cls = Mock(return_value=mock_solver)

        pred = cfm_instance.infer(step_data, return_trajectory=True, num_steps=10)

        # With single sample, _aggregate_predictions adds a sample dimension: traj becomes (10,1,4,2)
        # X should be last step, shape (4,2)
        assert pred.X.shape == (4, 2)
        # traj should be (10,1,4,2)
        assert pred.traj is not None
        assert pred.traj.shape == (10, 1, 4, 2)
        # raw_samples is None (single sample)
        assert pred.raw_samples is None

    def test_predict_with_trajectory_multiple_samples(self, cfm_instance):
        step_data = Mock()
        step_data.target_state = torch.randn(4, 2)
        step_data.source_state = None  # generate from noise
        step_data.target_condition_data = mock_condition_data()
        step_data.target_group_data = mock_condition_data()
        mock_solver = Mock()
        # Simulate trajectory for n_samples=3: solver returns (num_steps, n_samples*batch, dim)
        n_samples = 3
        batch = 4
        dim = 2
        n_steps = 10
        traj_flat = torch.randn(n_steps, n_samples * batch, dim)
        mock_solver.solve.return_value = traj_flat

        cfm_instance._default_solver_cls = Mock(return_value=mock_solver)

        pred = cfm_instance.infer(step_data, return_trajectory=True, num_steps=n_steps, n_samples=n_samples)

        # After _aggregate_predictions:
        # X = averaged over samples -> shape (batch, dim)
        assert pred.X.shape == (batch, dim)
        assert pred.X.ndim == 2
        assert pred.traj is not None
        assert pred.traj.ndim >= 3
        # raw_samples should be (n_samples, batch, dim)
        assert pred.raw_samples is not None
        assert pred.raw_samples.shape == (n_samples, batch, dim)

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
        cfm_instance.infer(step_data, solver_cls=mock_solver_cls, num_steps=5)
        mock_solver_cls.assert_called_once()
        call_kwargs = mock_solver_cls.call_args[1]
        assert call_kwargs.get("method") == "euler"

    # -------------------------------------------------------------------------
    # Compatibility with base class methods
    # -------------------------------------------------------------------------
    def test_train_step(self, cfm_instance):
        matched_distr = Mock()
        step_data = StepData(
            target_state=torch.randn(4, 2),
            target_coupling_lin=torch.randn(4, 2),
            target_coupling_quad=None,
            target_condition_data=torch.randn(4, 3),  # tensor, not dict
            target_group_data=torch.randn(4, 3),  # tensor
            source_state=torch.randn(4, 2),
            source_coupling_lin=torch.randn(4, 2),
            source_coupling_quad=None,
            source_condition_data=torch.randn(4, 3),  # tensor
            source_group_data=torch.randn(4, 3),  # tensor
        )

        with patch("sckitflow.core.methods._base.extract_step_data", return_value=step_data):
            cfm_instance._match_fn = lambda source_lin, target_lin, source_quad, target_quad: (
                np.arange(4),
                np.arange(4),
            )
            cfm_instance.compute_loss = Mock(return_value=(torch.tensor(0.5), {"loss": 0.5}))

            loss, log_dict = cfm_instance.train_step(matched_distr)

            assert loss == torch.tensor(0.5)
            assert log_dict == {"loss": 0.5}
