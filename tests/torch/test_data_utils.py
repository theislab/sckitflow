from unittest.mock import Mock

import numpy as np
import pytest
import torch

from sckitflow.core import _data_utils as data_utils
from sckitflow.core._types import StepData
from sckitflow.data._composite import MatchedDistributions
from sckitflow.data._mixins import BatchMixin
from sckitflow.data.containers import CouplingData, DistributionData, StateData


@pytest.fixture
def batch_mixin():
    batch_mixin = Mock(spec=BatchMixin)
    batch_mixin.mapping = {"a": np.array([1, 2]), "b": np.array([3.0, 4.0])}
    return batch_mixin


@pytest.fixture
def state_data():
    state_data = Mock(spec=StateData)
    state_data.X = np.random.randn(5, 2)
    return state_data


@pytest.fixture
def coupling_data():
    coupling = Mock(spec=CouplingData)
    coupling.state_lin = Mock()
    coupling.state_lin.X = np.random.randn(3, 2)
    coupling.state_quad = None
    return coupling


@pytest.fixture
def distribution_data():
    state_data = StateData(X=np.random.randn(5, 2))
    dist_data = Mock(spec=DistributionData)
    dist_data.state_data = state_data
    dist_data.condition_data = Mock()
    dist_data.groups_data = Mock()

    # The critical fix: give state_lin a real .X array
    coupling = Mock()
    coupling.state_lin = Mock()
    coupling.state_lin.X = np.random.randn(3, 2)
    coupling.state_quad = None
    dist_data.source_coupling_data = coupling

    return dist_data


@pytest.fixture
def matched_data():
    matched = Mock(spec=MatchedDistributions)
    source_dist = Mock(spec=DistributionData)
    target_dist = Mock(spec=DistributionData)
    matched.source_distribution = source_dist
    matched.target_distribution = target_dist
    return matched


class TestDataUtils:
    def test_batchmixin_to_torch(self, batch_mixin):
        result = data_utils.batchmixin_to_torch(batch_mixin, dtype=torch.float32)
        assert isinstance(result["a"], torch.Tensor)
        assert result["a"].dtype == torch.float32
        assert result["a"].tolist() == [1, 2]
        assert result["b"].tolist() == [3.0, 4.0]

    def test_extract_state_data(self, state_data):
        tensor = data_utils.extract_state_data(state_data)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (5, 2)
        assert data_utils.extract_state_data(None) is None

    def test_extract_coupling_data(self, distribution_data):
        lin, quad = data_utils.extract_coupling_data(distribution_data, "source")
        assert lin is not None
        assert quad is None

    def test_extract_distribution_data(self, distribution_data):
        result = data_utils.extract_distribution_data(distribution_data, "source")
        assert len(result) == 5

    def test_extract_step_data(self):
        # Set up mocks with the required coupling data for both source and target
        matched = Mock(spec=MatchedDistributions)

        source_dist = Mock(spec=DistributionData)
        target_dist = Mock(spec=DistributionData)

        # Source coupling
        src_coupling = Mock()
        src_coupling.state_lin = Mock()
        src_coupling.state_lin.X = np.random.randn(3, 2)
        src_coupling.state_quad = None
        source_dist.source_coupling_data = src_coupling

        # Target coupling
        tgt_coupling = Mock()
        tgt_coupling.state_lin = Mock()
        tgt_coupling.state_lin.X = np.random.randn(4, 2)
        tgt_coupling.state_quad = None
        target_dist.target_coupling_data = tgt_coupling

        # Other required attributes
        source_dist.state_data = StateData(X=np.random.randn(5, 2))
        source_dist.condition_data = Mock()
        source_dist.groups_data = Mock()
        target_dist.state_data = StateData(X=np.random.randn(6, 2))
        target_dist.condition_data = Mock()
        target_dist.groups_data = Mock()

        matched.source_distribution = source_dist
        matched.target_distribution = target_dist

        step_data = data_utils.extract_step_data(matched)
        assert isinstance(step_data, StepData)

    def test_write_continuous_cond_cov_no_base_data(self):
        """When base_data is None, a fresh StepData with only the condition is returned."""
        key = "my_cov"
        x = torch.randn(10, 5)
        result = data_utils.write_continuous_cond_cov_to_step_data(key, x)

        assert isinstance(result, StepData)
        assert result.target_condition_data == {key: x}
        # All other fields should be the default None
        assert result.target_state is None
        assert result.source_state is None
        assert result.target_coupling_lin is None
        assert result.source_coupling_lin is None
        assert result.target_group_data is None
        assert result.source_condition_data is None

    @pytest.mark.xfail(
        reason="https://github.com/theislab/sckitflow/issues/146 - StepData carries a plain dict where MixedTypeData is expected",
        strict=True,
    )
    def test_write_continuous_cond_cov_with_base_data(self, monkeypatch):
        """When base_data is provided, the existing StepData is updated with the new condition."""
        key = "new_cov"
        x = torch.randn(10, 5)

        # Build a realistic StepData that extract_step_data would return
        existing_step = StepData(
            target_state=torch.randn(5, 2),
            target_coupling_lin=torch.randn(5, 2),
            target_coupling_quad=None,
            target_condition_data={"old_key": torch.randn(5, 3)},
            target_group_data=None,
            source_state=torch.randn(5, 2),
            source_coupling_lin=torch.randn(5, 2),
            source_coupling_quad=None,
            source_condition_data={"old_key": torch.randn(5, 3)},
            source_group_data=None,
        )

        base_data = Mock(spec=MatchedDistributions)  # dummy, won't be used directly
        # Patch extract_step_data to return our prepared StepData
        monkeypatch.setattr(data_utils, "extract_step_data", lambda *a, **kw: existing_step)

        result = data_utils.write_continuous_cond_cov_to_step_data(
            key, x, base_data, dtype=torch.float64, device=torch.device("cpu")
        )

        # Original condition still present
        assert torch.equal(result.target_condition_data["old_key"], existing_step.target_condition_data["old_key"])
        # New condition inserted
        assert torch.equal(result.target_condition_data[key], x)
        # Other fields untouched
        assert torch.equal(result.target_state, existing_step.target_state)
        assert torch.equal(result.source_state, existing_step.source_state)
        assert torch.equal(result.target_coupling_lin, existing_step.target_coupling_lin)
        # Source condition data unchanged (the new key should not appear there)
        assert key not in result.source_condition_data
        assert torch.equal(result.source_condition_data["old_key"], existing_step.source_condition_data["old_key"])

    def test_prepare_latent_train_generate_from_noise(self):
        target = torch.randn(4, 2)
        latent = data_utils.prepare_latent_train(None, target, torch.randn, generate_from_noise=True)
        assert latent.shape == target.shape
        assert not torch.allclose(latent, target)

    def test_prepare_latent_train_use_source(self):
        source = torch.randn(4, 2)
        target = torch.randn(4, 2)
        latent = data_utils.prepare_latent_train(source, target, torch.randn, generate_from_noise=False)
        assert latent is source

    def test_prepare_latent_inference_single_sample(self):
        target = torch.randn(4, 2)
        latent = data_utils.prepare_latent_inference(
            None, target, torch.randn, n_samples=None, generate_from_noise=True
        )
        assert latent.shape == (4, 2)

    def test_prepare_latent_inference_multiple_samples(self):
        target = torch.randn(4, 2)
        latent = data_utils.prepare_latent_inference(None, target, torch.randn, n_samples=3, generate_from_noise=True)
        assert latent.shape == (3, 4, 2)

    def test_prepare_latent_inference_source_no_generation(self):
        source = torch.randn(4, 2)
        target = torch.randn(4, 2)
        latent = data_utils.prepare_latent_inference(
            source, target, torch.randn, n_samples=5, generate_from_noise=False
        )
        assert latent is source
        assert latent.shape == (4, 2)  # unchanged
