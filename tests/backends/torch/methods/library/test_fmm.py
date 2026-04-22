from unittest.mock import Mock, patch

import pytest
import torch

from sc_flow.backends.torch._types import PredictionData
from sc_flow.backends.torch.methods._utils import StepData
from sc_flow.backends.torch.methods.library._fmm import FMM
from sc_flow.data.containers._mixed_type import MixedTypeData


# -----------------------------------------------------------------------------
# Dummy module that replaces MLPFlowMap for testing
# -----------------------------------------------------------------------------
class DummyModule(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)

    def forward(self, s, t, x, condition_dict=None, source=None):
        return self.linear(x)

    def get_vf_fn(self, condition_dict=None, source=None):
        def vf_fn(s, t, x):
            return self.linear(x)

        return vf_fn

    @classmethod
    def init_from_dims_registry(cls, dims_registry, *args, **kwargs):
        return cls(*args, **kwargs)


# Helper for condition data
def mock_condition_data():
    mock_data = Mock(spec=MixedTypeData)
    mock_reps = Mock()
    mock_reps.mapping = {}
    mock_data.extract_reps.return_value = mock_reps
    return mock_data


# -----------------------------------------------------------------------------
# Fixture
# -----------------------------------------------------------------------------
@pytest.fixture
def fmm_instance():
    dims_reg = Mock()
    dims_reg.feature_names = ["g1", "g2"]
    dims_reg.n_features = 2
    dm = Mock()

    original_module_cls = FMM._module_cls
    FMM._module_cls = DummyModule

    # Patch _extract_matched_observations to bypass matching logic
    with patch.object(FMM, "_extract_matched_observations", side_effect=lambda x: x):
        fmm = FMM(dims_registry=dims_reg, dm=dm, is_paired_setting=False, dtype=torch.float32, device_id="cpu")

    FMM._module_cls = original_module_cls
    fmm._device_id = "cpu"
    return fmm


# -----------------------------------------------------------------------------
# Test suite
# -----------------------------------------------------------------------------
class TestFMM:
    """Tests for Flow Map Matching (FMM) class."""

    def test_init_defaults(self, fmm_instance):
        assert fmm_instance._match_fn is not None
        assert fmm_instance._noise_sampler is not None
        assert fmm_instance._probability_path is not None
        assert fmm_instance._time_sampler is not None
        s, t = fmm_instance._time_sampler((4,))
        assert s.shape == (4,)
        assert t.shape == (4,)

    def test_init_with_teacher(self):
        dims_reg = Mock()
        dm = Mock()
        teacher = Mock()
        teacher.match_fn = Mock()
        teacher.noise_sampler = Mock()
        teacher.probability_path = Mock()

        with (
            patch.object(FMM, "_module_cls", DummyModule),
            patch.object(FMM, "_extract_matched_observations", side_effect=lambda x: x),
        ):
            fmm = FMM(dims_reg, dm, False, cfm=teacher)
            assert fmm._match_fn is teacher.match_fn
            assert fmm._noise_sampler is teacher.noise_sampler
            assert fmm._probability_path is teacher.probability_path
            assert fmm._cfm is teacher

    def test_prepare_latent_state(self, fmm_instance):
        fmm_instance._generate_from_noise = True
        target = torch.randn(4, 2)
        latent = fmm_instance._prepare_latent_state(None, target)
        assert latent.shape == (4, 2)
        fmm_instance._generate_from_noise = False
        source = torch.randn(4, 2)
        latent = fmm_instance._prepare_latent_state(source, target)
        assert latent is source

    def test_compute_loss_e2e(self, fmm_instance):
        step_data = StepData(
            target_state=torch.randn(4, 2),
            target_coupling_lin=None,
            target_coupling_quad=None,
            target_condition_data=mock_condition_data(),
            target_group_data=mock_condition_data(),
            source_state=torch.randn(4, 2),
            source_coupling_lin=None,
            source_coupling_quad=None,
            source_condition_data=mock_condition_data(),
            source_group_data=mock_condition_data(),
        )
        fmm_instance._probability_path = Mock()
        fmm_instance._probability_path.compute_xt.return_value = torch.randn(4, 2)
        fmm_instance._probability_path.compute_ut.return_value = torch.randn(4, 2)
        fmm_instance._module.forward = Mock(return_value=torch.randn(4, 2))
        fmm_instance._cfm = None
        fmm_instance._time_sampler = Mock(return_value=(torch.rand(4), torch.rand(4)))
        fmm_instance._weight_fn = lambda s, t: torch.ones_like(s)

        loss, log = fmm_instance._compute_loss_e2e(step_data)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert "loss" in log

    def test_compute_loss_distillation(self, fmm_instance):
        step_data = StepData(
            target_state=torch.randn(4, 2),
            target_coupling_lin=None,
            target_coupling_quad=None,
            target_condition_data=mock_condition_data(),
            target_group_data=mock_condition_data(),
            source_state=torch.randn(4, 2),
            source_coupling_lin=None,
            source_coupling_quad=None,
            source_condition_data=mock_condition_data(),
            source_group_data=mock_condition_data(),
        )
        teacher = Mock()
        teacher._module.get_vf_fn.return_value = lambda t, x: torch.randn(4, 2)
        fmm_instance._cfm = teacher
        fmm_instance._probability_path = Mock()
        fmm_instance._probability_path.compute_xt.return_value = torch.randn(4, 2)
        fmm_instance._module.get_vf_fn = Mock(return_value=Mock())
        fmm_instance._time_sampler = Mock(return_value=(torch.rand(4), torch.rand(4)))
        fmm_instance._weight_fn = lambda s, t: torch.ones_like(s)

        with patch("torch.func.jvp") as mock_jvp:
            mock_jvp.return_value = (torch.randn(4, 2), torch.randn(4, 2))
            loss, log = fmm_instance._compute_loss_distillation(step_data)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert "loss" in log

    def test_compute_loss_dispatches(self, fmm_instance):
        step_data = Mock()
        fmm_instance._cfm = None
        with patch.object(fmm_instance, "_compute_loss_e2e", return_value=(torch.tensor(0.1), {})) as mock_e2e:
            loss, _ = fmm_instance._compute_loss(step_data)
            mock_e2e.assert_called_once()
            assert loss == 0.1
        teacher = Mock()
        fmm_instance._cfm = teacher
        with patch.object(
            fmm_instance, "_compute_loss_distillation", return_value=(torch.tensor(0.2), {})
        ) as mock_dist:
            loss, _ = fmm_instance._compute_loss(step_data)
            mock_dist.assert_called_once()
            assert loss == 0.2

    def test_predict_no_trajectory(self, fmm_instance):
        step_data = Mock()
        step_data.target_state = torch.randn(4, 2)
        step_data.source_state = torch.randn(4, 2)
        step_data.target_condition_data = mock_condition_data()
        step_data.target_group_data = mock_condition_data()
        fmm_instance._prepare_latent_state = Mock(return_value=torch.randn(4, 2))
        fmm_instance._module.get_vf_fn = Mock(return_value=lambda s, t, x: x)
        pred = fmm_instance._predict(step_data, return_trajectory=False, num_steps=3)
        assert isinstance(pred, PredictionData)
        assert pred.samples is not None
        assert pred.traj is None
        assert pred.samples.shape == (4, 2)

    def test_predict_with_trajectory(self, fmm_instance):
        step_data = Mock()
        step_data.target_state = torch.randn(4, 2)
        step_data.source_state = torch.randn(4, 2)
        step_data.target_condition_data = mock_condition_data()
        step_data.target_group_data = mock_condition_data()
        fmm_instance._prepare_latent_state = Mock(return_value=torch.randn(4, 2))
        fmm_instance._module.get_vf_fn = Mock(return_value=lambda s, t, x: x)
        pred = fmm_instance._predict(step_data, return_trajectory=True, num_steps=3)
        assert isinstance(pred, PredictionData)
        assert pred.samples is not None
        assert pred.traj is not None
        # With 3 steps we get 4 states (initial + 3 steps)
        assert pred.traj.shape == (4, 4, 2)
        assert torch.equal(pred.samples, pred.traj[-1])

    def test_train_step(self, fmm_instance):
        matched_distr = Mock()
        step_data = Mock()
        fmm_instance._extract_step_data = Mock(return_value=step_data)
        fmm_instance._compute_loss = Mock(return_value=(torch.tensor(0.5), {"loss": 0.5}))
        loss, log_dict = fmm_instance.train_step(matched_distr)
        assert loss == torch.tensor(0.5)
        assert log_dict == {"loss": 0.5}
