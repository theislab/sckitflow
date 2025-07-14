import pytest
import torch

from sc_flow.backends.torch.probability_paths._probability_paths import (
    BaseProbabilityPath,
    LinearDiracProbabilityPath,
    LinearGaussianProbabilityPath,
    SchrodingerBridgeProbabilityPath,
    VariancePreservingDiracProbabilityPath,
)

from .utils import verify_method_output

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
        # non deterministic probability paths
        if not probability_path_cls.is_deterministic:
            # initialize with negative sigma
            with pytest.raises(ValueError, match=r"Argument sigma should be a positive float"):
                probability_path = probability_path_cls(-1.0, prng=torch.random.default_generator)
                return None

            # initialize with prng
            probability_path = probability_path_cls(1.0, prng=torch.random.default_generator)
            assert not probability_path.is_deterministic
        else:
            # initialize with prng (should only raise a warning)
            probability_path = probability_path_cls(prng=torch.random.default_generator)
            assert probability_path.is_deterministic

            # initialize without prng
            probability_path = probability_path_cls(prng=None)
            assert probability_path.is_deterministic

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
        probability_path = probability_path_cls(
            1.0, prng=None if probability_path_cls.is_deterministic else torch.random.default_generator
        )
        verify_method_output(
            probability_path,
            method,
            batch_size,
            num_feats,
            num_channels,
            height,
            width,
        )
