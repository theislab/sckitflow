import pytest
import torch

from sc_flow.backends.torch.probability_paths._probability_paths import (
    BaseProbabilityPath,
    LinearDiracProbabilityPath,
    LinearGaussianProbabilityPath,
    SchrodingerBridgeProbabilityPath,
    VariancePreservingDiracProbabilityPath,
)

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
        tested_method = getattr(probability_path, method)

        # 2D - case 0: correct inputs
        t = torch.zeros((batch_size, 1))
        x0 = torch.zeros((batch_size, num_feats))
        x1 = torch.zeros((batch_size, num_feats))
        if method == "compute_ut":
            xt = torch.zeros((batch_size, num_feats))
            out = tested_method(t, xt, x0, x1)
        else:
            out = tested_method(t, x0, x1)
        assert out.shape == (batch_size, num_feats)

        # 2D - case 1: shape mismatch between x0 and x1
        t = torch.zeros((batch_size, 1))
        x0 = torch.zeros((batch_size, num_feats))
        x1 = torch.zeros((batch_size, num_feats + 1))
        with pytest.raises(ValueError, match=r"`input_tensor` and `target_tensor` are supposed to have the same shape"):
            if method == "compute_ut":
                xt = torch.zeros((batch_size, num_feats))
                out = tested_method(t, xt, x0, x1)
            else:
                out = tested_method(t, x0, x1)

        # 3D - case 0: correct inputs
        t = torch.zeros((batch_size, 1))
        x0 = torch.zeros((batch_size, num_channels, height, width))
        x1 = torch.zeros((batch_size, num_channels, height, width))
        if method == "compute_ut":
            xt = torch.zeros((batch_size, num_channels, height, width))
            out = tested_method(t, xt, x0, x1)
        else:
            out = tested_method(t, x0, x1)
        assert out.shape == (batch_size, num_channels, height, width)

        # 3D - case 1: shape mismatch between x0 and x1
        t = torch.zeros((batch_size, 1))
        x0 = torch.zeros((batch_size, num_channels, height, width))
        x1 = torch.zeros((batch_size, num_channels, height, width + 1))
        with pytest.raises(ValueError, match=r"`input_tensor` and `target_tensor` are supposed to have the same shape"):
            if method == "compute_ut":
                xt = torch.zeros((batch_size, num_channels, height, width))
                out = tested_method(t, xt, x0, x1)
            else:
                out = tested_method(t, x0, x1)
