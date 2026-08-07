from unittest.mock import Mock

import numpy as np
import pytest
import torch

from sckitflow.core import _data_utils as data_utils
from sckitflow.data._mixins import BatchMixin


@pytest.fixture
def batch_mixin():
    batch_mixin = Mock(spec=BatchMixin)
    batch_mixin.mapping = {"a": np.array([1, 2]), "b": np.array([3.0, 4.0])}
    return batch_mixin


class TestDataUtils:
    def test_batchmixin_to_torch(self, batch_mixin):
        result = data_utils.batchmixin_to_torch(batch_mixin, dtype=torch.float32)
        assert isinstance(result["a"], torch.Tensor)
        assert result["a"].dtype == torch.float32
        assert result["a"].tolist() == [1, 2]
        assert result["b"].tolist() == [3.0, 4.0]

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
