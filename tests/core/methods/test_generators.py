import numpy as np
import pandas as pd
import pytest
import torch

from sckitflow._random import generators
from sckitflow.core.methods.training._cfm import CFMTraining, CFMTrainingConfig
from sckitflow.core.probability_paths._config import LinearTwoSidedPathConfig
from sckitflow.data._dims import DataDimensions


class _Field(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)

    def forward(self, t, xt, condition_dict=None, source=None):
        return self.linear(xt)


@pytest.fixture
def dims() -> DataDimensions:
    """Shape spec matching `_Field`'s `Linear(3, 3)`.

    The shared conftest fixture uses `state_dim=4`; `_Field` was built for 3, so
    the training latent has to be 3 as well.
    """
    return DataDimensions(
        state_dim=3,
        feature_names=pd.Index([f"f{i}" for i in range(3)]),
        condition_reps_dims=None,
        condition_continuous_dims=None,
        response_reps_dims=None,
        response_continuous_dims=None,
        groups_reps_dims=None,
        source_lin_dim=None,
        source_quad_dim=None,
        target_lin_dim=None,
        target_quad_dim=None,
    )


def _loss(method: CFMTraining, seed: int) -> torch.Tensor:
    batch = {
        "target_state": torch.ones(8, 3),
        "source_state": None,
        "target_condition_data": None,
        "target_group_data": None,
    }
    return method.compute_loss(batch, generator=generators(seed, 0))[0]


def test_loss_depends_only_on_the_passed_generator(dims):
    method = CFMTraining(
        _Field(),
        CFMTrainingConfig(probability_path=LinearTwoSidedPathConfig(sigma=0.1)),
        dims=dims,
    )
    first = _loss(method, seed=0)
    torch.manual_seed(123)  # the global RNG must not matter
    np.random.seed(123)  # noqa: NPY002  (checking it is ignored)
    assert torch.equal(first, _loss(method, seed=0))
    assert not torch.equal(first, _loss(method, seed=1))


def test_streams_are_independent_and_repeatable():
    a = generators(0, 0, 5)
    b = generators(0, 0, 5)
    c = generators(0, 0, 6)
    x, y, z = (torch.rand(4, generator=g) for g in (a, b, c))
    assert torch.equal(x, y) and not torch.equal(x, z)
