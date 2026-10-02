import pytest
import torch

from sckitflow._runtime import set_backend
from sckitflow.core.probability_paths._probability_paths import (
    BaseProbabilityPath,
    LinearDiracProbabilityPath,
    LinearGaussianProbabilityPath,
    SchrodingerBridgeProbabilityPath,
    VariancePreservingDiracProbabilityPath,
)

from ...utils import verify_method_output  # noqa

batch_size = 8
num_feats = 16
num_channels = 3
height = 32
width = 64


class TestProbabilityPaths:
    @pytest.mark.parametrize(
        "probability_path_cls",
        [
            LinearGaussianProbabilityPath,
            SchrodingerBridgeProbabilityPath,
            LinearDiracProbabilityPath,
            VariancePreservingDiracProbabilityPath,
        ],
    )
    def test_probability_path_init(
        self,
        probability_path_cls: type[BaseProbabilityPath],
    ) -> None:
        set_backend("torch")

        if not probability_path_cls.is_deterministic:
            with pytest.raises(ValueError, match=r"Argument sigma should be a positive float"):
                probability_path_cls(-1.0)
            assert not probability_path_cls(1.0).is_deterministic
        else:
            assert probability_path_cls().is_deterministic

    @pytest.mark.parametrize(
        "probability_path_cls",
        [
            LinearGaussianProbabilityPath,
            SchrodingerBridgeProbabilityPath,
            LinearDiracProbabilityPath,
            VariancePreservingDiracProbabilityPath,
        ],
    )
    @pytest.mark.parametrize("method", ["compute_xt", "compute_mu_t", "compute_ut"])
    def test_probability_path_methods(
        self,
        probability_path_cls: BaseProbabilityPath,
        method: str,
    ) -> None:
        set_backend("torch")

        # initialize probability path and retrieving method to test
        probability_path = probability_path_cls(1.0)
        verify_method_output(
            probability_path,
            method,
            batch_size,
            num_feats,
            num_channels,
            height,
            width,
        )


@pytest.mark.parametrize("probability_path_cls", [LinearGaussianProbabilityPath, SchrodingerBridgeProbabilityPath])
def test_noise_comes_only_from_the_generator(probability_path_cls: type[BaseProbabilityPath]) -> None:
    path, t, x0, x1 = probability_path_cls(1.0), torch.full((4, 1), 0.5), torch.zeros(4, 3), torch.ones(4, 3)
    same = [path.compute_xt(t, x0, x1, generator=torch.Generator().manual_seed(0)) for _ in range(2)]
    torch.manual_seed(1)  # the global RNG must not matter
    other = path.compute_xt(t, x0, x1, generator=torch.Generator().manual_seed(1))
    assert torch.equal(same[0], same[1]) and not torch.equal(same[0], other)
