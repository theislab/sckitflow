import pytest

from sc_flow._runtime import set_backend
from sc_flow.backends.jax.probability_paths._probability_paths import (
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
        set_backend("jax")

        if not probability_path_cls.is_deterministic:
            with pytest.raises(ValueError, match=r"Argument sigma should be a positive float"):
                probability_path = probability_path_cls(-1.0)
                return None

            probability_path = probability_path_cls(1.0)
            assert not probability_path.is_deterministic
        else:
            probability_path = probability_path_cls()
            assert probability_path.is_deterministic

            probability_path = probability_path_cls()
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
        set_backend("jax")
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
