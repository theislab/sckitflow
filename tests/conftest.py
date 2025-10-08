import anndata as ad
import numpy as np
import pytest
import torch

from .utils import get_dummy_network

input_dim = 10
output_dim = 10
hidden_dims = (20, 20)
batch_size = 32


@pytest.fixture
def adata():
    adata = ad.AnnData(X=np.array([[1.2, 2.3], [3.4, 4.5], [5.6, 6.7]]).astype(np.float32))
    adata.layers["scaled"] = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]).astype(np.float32)

    return adata


@pytest.fixture
def dummy_method(input_di=input_dim, output_dim=output_dim, hidden_dims=hidden_dims):
    return get_dummy_network(input_dim, output_dim, hidden_dims)


@pytest.fixture
def dummy_callbacks():
    class DummyCallbacks:
        def __init__(self):
            self.called_with = []

        def run_on_valid_step(self, validation_dict, condition):
            self.called_with.append((validation_dict, condition))
            return {"val_loss": 0.456}

    return DummyCallbacks()


@pytest.fixture
def dummy_trainloader():
    class DummyTrainLoader:
        def __init__(self):
            self.sample_calls = 0

        def sample(self, _):
            self.sample_calls += 1
            return {"source": torch.rand((batch_size, output_dim)), "target": torch.rand((batch_size, output_dim))}

    return DummyTrainLoader()


@pytest.fixture
def dummy_valloader():
    class DummyValLoader:
        def __init__(self):
            self.sample_calls = 0

        def sample(self, _):
            self.sample_calls += 1
            return {
                "condA": {
                    "source": torch.rand((batch_size, output_dim)),
                    "target": torch.rand((batch_size, output_dim)),
                }
            }

    return DummyValLoader()
