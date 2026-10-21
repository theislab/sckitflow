import inspect

import pytest
import torch

from sckitflow.core import probability_paths
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


ALL_PATHS = [
    cls
    for name in probability_paths.__all__
    if isinstance(cls := getattr(probability_paths, name), type)
    and issubclass(cls, BaseProbabilityPath)
    and not inspect.isabstract(cls)
]


def _path_and_data(cls: type[BaseProbabilityPath]) -> tuple[BaseProbabilityPath, torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    x0, x1 = torch.randn(5, 3, generator=g, dtype=torch.float64), torch.randn(5, 3, generator=g, dtype=torch.float64)
    return cls(0.7) if not cls.is_deterministic else cls(), x0, x1


def test_all_paths_collected() -> None:
    assert len(ALL_PATHS) == 4


@pytest.mark.parametrize("probability_path_cls", ALL_PATHS)
def test_mean_hits_endpoints(probability_path_cls: type[BaseProbabilityPath]) -> None:
    path, x0, x1 = _path_and_data(probability_path_cls)
    t = torch.ones(5, 1, dtype=torch.float64)
    torch.testing.assert_close(path.compute_mu_t(0 * t, x0, x1), x0)
    torch.testing.assert_close(path.compute_mu_t(t, x0, x1), x1)


@pytest.mark.parametrize("probability_path_cls", ALL_PATHS)
def test_velocity_generates_path(probability_path_cls: type[BaseProbabilityPath]) -> None:
    path, x0, x1 = _path_and_data(probability_path_cls)
    eps = torch.randn(5, 3, generator=torch.Generator().manual_seed(1), dtype=torch.float64)

    def xt_of(t: torch.Tensor) -> torch.Tensor:
        t = t.expand(5, 3)
        return path.compute_mu_t(t, x0, x1) + path.compute_sigma_t(t) * eps

    t = torch.tensor([[0.1], [0.3], [0.5], [0.7], [0.9]], dtype=torch.float64)
    xt, dxt_dt = torch.func.jvp(xt_of, (t,), (torch.ones_like(t),))
    torch.testing.assert_close(path.compute_ut(t, xt, x0, x1), dxt_dt)
