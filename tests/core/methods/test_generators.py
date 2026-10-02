import numpy as np
import torch

from sckitflow._random import generators
from sckitflow.core.methods.training._cfm import CFMTraining, CFMTrainingConfig
from sckitflow.core.probability_paths._probability_paths import LinearGaussianProbabilityPath


class _Field(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)

    def forward(self, t, xt, condition_dict=None, source=None):
        return self.linear(xt)


def _loss(method: CFMTraining, seed: int) -> torch.Tensor:
    batch = {
        "target_state": torch.ones(8, 3),
        "source_state": None,
        "target_condition_data": None,
        "target_group_data": None,
    }
    generator, rng = generators(seed, 0)
    return method.compute_loss(batch, generator=generator, rng=rng)[0]


def test_loss_depends_only_on_the_passed_generator():
    method = CFMTraining(_Field(), CFMTrainingConfig(probability_path=LinearGaussianProbabilityPath(sigma=0.1)))
    first = _loss(method, seed=0)
    torch.manual_seed(123)  # the global RNG must not matter
    np.random.seed(123)  # noqa: NPY002  (checking it is ignored)
    assert torch.equal(first, _loss(method, seed=0))
    assert not torch.equal(first, _loss(method, seed=1))


def test_streams_are_independent_and_repeatable():
    a, _ = generators(0, 0, 5)
    b, _ = generators(0, 0, 5)
    c, _ = generators(0, 0, 6)
    x, y, z = (torch.rand(4, generator=g) for g in (a, b, c))
    assert torch.equal(x, y) and not torch.equal(x, z)
