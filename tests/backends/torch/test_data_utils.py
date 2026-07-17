from unittest.mock import Mock

import numpy as np
import pytest
import torch

from sc_flow.backends.torch import data_utils
from sc_flow.backends.torch._types import StepData
from sc_flow.data._composite import MatchedDistributions
from sc_flow.data._mixins import BatchMixin
from sc_flow.data.containers import CouplingData, DistributionData, StateData


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
