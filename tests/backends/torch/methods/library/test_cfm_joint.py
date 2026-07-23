from unittest.mock import Mock

import numpy as np
import pytest
import torch

from sc_flow.backends.torch._types import PredictionData
from sc_flow.backends.torch.methods._utils import StepData
from sc_flow.backends.torch.methods.library._cfm_joint import CFM_Joint


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


class DummyMixedTypeData:
    def __getitem__(self, idx):
        return self

    def extract_reps(self):
        return Mock(mapping={})


def mock_condition_data():
    return DummyMixedTypeData()


def _make_step_data(batch_size=4, dim=2):
    """Create a StepData with known tensors."""
    mock_cond = mock_condition_data()
    mock_group = mock_condition_data()
    return StepData(
        target_state=torch.randn(batch_size, dim),
        target_coupling_lin=torch.randn(batch_size, dim),
        target_coupling_quad=None,
        target_condition_data=mock_cond,
        target_group_data=mock_group,
        source_state=torch.randn(batch_size, dim),
        source_coupling_lin=torch.randn(batch_size, dim),
        source_coupling_quad=None,
        source_condition_data=mock_cond,
        source_group_data=mock_group,
    )


# -----------------------------------------------------------------------------
# Fixture
# -----------------------------------------------------------------------------
@pytest.fixture
def cfm_joint_instance():
    dims_reg = Mock()
    dims_reg.feature_names = ["g1", "g2"]
    dims_reg.n_features = 2
    dm = Mock()

    original_module_cls = CFM_Joint._module_cls
    CFM_Joint._module_cls = DummyModule

    cfm_j = CFM_Joint(
        dims_registry=dims_reg,
        dm=dm,
        is_paired_setting=False,
        dtype=torch.float32,
        device_id="cpu",
        timepoints=(0.0, 0.3, 0.7, 1.0),
    )

    CFM_Joint._module_cls = original_module_cls
    return cfm_j


# -----------------------------------------------------------------------------
# Init tests
# -----------------------------------------------------------------------------
class TestCFMJointInit:
    def test_init_requires_timepoints(self):
        """Raises ValueError when timepoints is None."""
        dims_reg = Mock()
        dims_reg.feature_names = ["g1", "g2"]
        dims_reg.n_features = 2
        dm = Mock()

        original = CFM_Joint._module_cls
        CFM_Joint._module_cls = DummyModule
        try:
            with pytest.raises(ValueError, match="CFM-J requires"):
                CFM_Joint(
                    dims_registry=dims_reg, dm=dm, is_paired_setting=False,
                    dtype=torch.float32, device_id="cpu", timepoints=None,
                )
        finally:
            CFM_Joint._module_cls = original

    def test_init_stores_timepoints_as_tuple(self, cfm_joint_instance):
        """Stores timepoints as a tuple with correct values."""
        assert cfm_joint_instance.timepoints == (0.0, 0.3, 0.7, 1.0)
        assert isinstance(cfm_joint_instance.timepoints, tuple)

    def test_init_accepts_list(self):
        """Converts a list of timepoints to a tuple."""
        dims_reg = Mock()
        dims_reg.feature_names = ["g1", "g2"]
        dims_reg.n_features = 2
        dm = Mock()

        original = CFM_Joint._module_cls
        CFM_Joint._module_cls = DummyModule
        try:
            cfm_j = CFM_Joint(
                dims_registry=dims_reg, dm=dm, is_paired_setting=False,
                dtype=torch.float32, device_id="cpu", timepoints=[0, 0.5, 1],
            )
            assert isinstance(cfm_j.timepoints, tuple)
            assert cfm_j.timepoints == (0, 0.5, 1)
        finally:
            CFM_Joint._module_cls = original

    def test_is_joint_returns_true(self, cfm_joint_instance):
        """is_joint property returns True for CFM_Joint."""
        assert cfm_joint_instance.is_joint is True

    def test_inherits_cfm_defaults(self, cfm_joint_instance):
        """Inherits base CFM attributes (match_fn, samplers, probability_path)."""
        assert cfm_joint_instance._match_fn is not None
        assert cfm_joint_instance._noise_sampler is not None
        assert cfm_joint_instance._time_sampler is not None
        assert cfm_joint_instance._probability_path is not None


# -----------------------------------------------------------------------------
# train_step_joint tests
# -----------------------------------------------------------------------------
class TestCFMJointTrainStepJoint:
    def _setup_mocks(self, cfm_j, n_transitions, batch_size=4, dim=2):
        """Set up mocks for train_step_joint.

        Returns the list of StepData objects created (one per transition).
        """
        step_datas = [_make_step_data(batch_size, dim) for _ in range(n_transitions)]
        call_count = [0]

        def mock_extract(node):
            idx = call_count[0]
            call_count[0] += 1
            return step_datas[idx]

        cfm_j._extract_step_data = Mock(side_effect=mock_extract)
        cfm_j._extract_matched_observations = Mock(side_effect=lambda sd: sd)
        cfm_j._match_fn = lambda src_l, tgt_l, src_q, tgt_q: (
            np.arange(batch_size), np.arange(batch_size)
        )
        return step_datas

    def test_node_count_mismatch_raises(self, cfm_joint_instance):
        """Raises ValueError when node count doesn't match n_transitions."""
        nodes = [Mock(), Mock()]
        with pytest.raises(ValueError, match="Expected 3 nodes"):
            cfm_joint_instance.train_step_joint(nodes)

    def test_returns_loss_and_dict(self, cfm_joint_instance):
        """Returns a scalar loss tensor and a dict with 'loss' key."""
        self._setup_mocks(cfm_joint_instance, n_transitions=3)
        nodes = [Mock() for _ in range(3)]

        loss, info = cfm_joint_instance.train_step_joint(nodes)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert "loss" in info
        assert isinstance(info["loss"], float)

    def test_single_forward_pass(self, cfm_joint_instance):
        """Calls the neural network exactly once on concatenated batch."""
        self._setup_mocks(cfm_joint_instance, n_transitions=3)
        cfm_joint_instance._module = Mock(return_value=torch.randn(12, 2))
        nodes = [Mock() for _ in range(3)]

        cfm_joint_instance.train_step_joint(nodes)

        cfm_joint_instance._module.assert_called_once()

    def test_concatenation_size(self, cfm_joint_instance):
        """Concatenated xt and t have shape (n_transitions * batch_size, ...)."""
        batch_size = 4
        self._setup_mocks(cfm_joint_instance, n_transitions=3, batch_size=batch_size)
        cfm_joint_instance._module = Mock(return_value=torch.randn(12, 2))
        nodes = [Mock() for _ in range(3)]

        cfm_joint_instance.train_step_joint(nodes)

        call_args = cfm_joint_instance._module.call_args
        t_arg, xt_arg = call_args[0]
        assert xt_arg.shape == (3 * batch_size, 2)
        assert t_arg.shape == (3 * batch_size,)

    def test_time_global_range(self):
        """Verify t_global = t_local * delta_t + t_start."""
        dims_reg = Mock()
        dims_reg.feature_names = ["g1", "g2"]
        dims_reg.n_features = 2
        dm = Mock()

        original = CFM_Joint._module_cls
        CFM_Joint._module_cls = DummyModule
        try:
            cfm_j = CFM_Joint(
                dims_registry=dims_reg, dm=dm, is_paired_setting=False,
                dtype=torch.float32, device_id="cpu",
                timepoints=(0.0, 0.4, 1.0),
            )
        finally:
            CFM_Joint._module_cls = original

        batch_size = 4
        self._setup_mocks(cfm_j, n_transitions=2, batch_size=batch_size)

        # Fix time sampler to return 0.5 for all samples
        cfm_j._time_sampler = lambda shape, **kw: torch.full(shape, 0.5)

        # Capture the t argument passed to module
        cfm_j._module = Mock(return_value=torch.randn(2 * batch_size, 2))
        nodes = [Mock(), Mock()]

        cfm_j.train_step_joint(nodes)

        t_arg = cfm_j._module.call_args[0][0]
        # Transition 0: t_global = 0.5 * 0.4 + 0.0 = 0.2
        # Transition 1: t_global = 0.5 * 0.6 + 0.4 = 0.7
        expected_t0 = torch.full((batch_size,), 0.2)
        expected_t1 = torch.full((batch_size,), 0.7)
        expected = torch.cat([expected_t0, expected_t1])
        torch.testing.assert_close(t_arg, expected)

    def test_velocity_scaling(self, cfm_joint_instance):
        """Calls compute_ut once per transition for velocity scaling."""
        batch_size = 4
        self._setup_mocks(cfm_joint_instance, n_transitions=3, batch_size=batch_size)

        raw_ut = torch.ones(batch_size, 2)
        cfm_joint_instance._probability_path = Mock()
        cfm_joint_instance._probability_path.compute_xt.return_value = torch.randn(batch_size, 2)
        cfm_joint_instance._probability_path.compute_ut.return_value = raw_ut.clone()

        cfm_joint_instance._module = Mock(return_value=torch.randn(3 * batch_size, 2))
        nodes = [Mock() for _ in range(3)]

        cfm_joint_instance.train_step_joint(nodes)

        # Check the ut target passed to MSE loss
        # timepoints = (0.0, 0.3, 0.7, 1.0) → deltas = [0.3, 0.4, 0.3]
        # compute_ut returns raw_ut each time, so scaled ut should be raw_ut / delta_t
        # can verify via the loss computation indirectly,
        # but we verify compute_ut was called 3 times
        assert cfm_joint_instance._probability_path.compute_ut.call_count == 3

    def test_loss_is_mse(self, cfm_joint_instance):
        """Loss equals MSE between module output and velocity scaled by 1/delta_t."""
        batch_size = 4
        self._setup_mocks(cfm_joint_instance, n_transitions=3, batch_size=batch_size)

        # Known module output
        vt = torch.ones(3 * batch_size, 2) * 2.0
        cfm_joint_instance._module = Mock(return_value=vt)

        # Known probability path outputs
        raw_ut = torch.ones(batch_size, 2)
        cfm_joint_instance._probability_path = Mock()
        cfm_joint_instance._probability_path.compute_xt.return_value = torch.randn(batch_size, 2)
        cfm_joint_instance._probability_path.compute_ut.return_value = raw_ut.clone()

        nodes = [Mock() for _ in range(3)]
        loss, info = cfm_joint_instance.train_step_joint(nodes)

        # Reconstruct expected: ut / delta_t for each transition
        # timepoints = (0.0, 0.3, 0.7, 1.0) → deltas = [0.3, 0.4, 0.3]
        deltas = [0.3, 0.4, 0.3]
        ut_scaled_parts = [raw_ut / d for d in deltas]
        ut_scaled = torch.cat(ut_scaled_parts)
        expected_loss = torch.nn.functional.mse_loss(vt, ut_scaled)

        torch.testing.assert_close(loss, expected_loss)


# -----------------------------------------------------------------------------
# Predict tests
# -----------------------------------------------------------------------------
class TestCFMJointPredict:
    def test_predict_custom_time_grid(self, cfm_joint_instance):
        """ODE solver receives time grid spanning [t_start, t_end]."""
        step_data = Mock()
        step_data.target_state = torch.randn(4, 2)
        step_data.source_state = torch.randn(4, 2)
        step_data.target_condition_data = mock_condition_data()
        step_data.target_group_data = mock_condition_data()

        mock_solver = Mock()
        mock_solver.solve.return_value = torch.randn(4, 2)
        mock_solver_cls = Mock(return_value=mock_solver)
        cfm_joint_instance._default_solver_cls = mock_solver_cls

        cfm_joint_instance._predict(
            step_data, return_trajectory=False, num_steps=10,
            t_start=0.3, t_end=0.7,
        )

        # Check the time_grid passed to solver.solve
        call_args = mock_solver.solve.call_args
        time_grid = call_args[0][1]  # second positional arg
        assert torch.isclose(time_grid[0], torch.tensor(0.3))
        assert torch.isclose(time_grid[-1], torch.tensor(0.7))

    def test_predict_default_backward_compatible(self, cfm_joint_instance):
        """Defaults to [0, 1] time grid when t_start/t_end not provided."""
        step_data = Mock()
        step_data.target_state = torch.randn(4, 2)
        step_data.source_state = torch.randn(4, 2)
        step_data.target_condition_data = mock_condition_data()
        step_data.target_group_data = mock_condition_data()

        mock_solver = Mock()
        mock_solver.solve.return_value = torch.randn(4, 2)
        mock_solver_cls = Mock(return_value=mock_solver)
        cfm_joint_instance._default_solver_cls = mock_solver_cls

        cfm_joint_instance._predict(step_data, return_trajectory=False, num_steps=10)

        call_args = mock_solver.solve.call_args
        time_grid = call_args[0][1]
        assert torch.isclose(time_grid[0], torch.tensor(0.0))
        assert torch.isclose(time_grid[-1], torch.tensor(1.0))

    def test_predict_returns_prediction_data(self, cfm_joint_instance):
        """Returns a PredictionData object with correct shape."""
        step_data = Mock()
        step_data.target_state = torch.randn(4, 2)
        step_data.source_state = torch.randn(4, 2)
        step_data.target_condition_data = mock_condition_data()
        step_data.target_group_data = mock_condition_data()

        mock_solver = Mock()
        mock_solver.solve.return_value = torch.randn(4, 2)
        cfm_joint_instance._default_solver_cls = Mock(return_value=mock_solver)

        pred = cfm_joint_instance._predict(
            step_data, return_trajectory=False, num_steps=10,
            t_start=0.0, t_end=0.5,
        )

        assert isinstance(pred, PredictionData)
        assert pred.X.shape == (4, 2)
