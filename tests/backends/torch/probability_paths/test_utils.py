from collections.abc import Callable

import pytest
import torch

from sc_flow._constants import MU_T_FN_KEY, SIGMA_T_FN_KEY, U_T_FN_KEY
from sc_flow._utils import get_fn_args_names_and_types
from sc_flow.backends.torch.probability_paths._probability_paths import (
    BaseProbabilityPath,
    LinearDiracProbabilityPath,
    LinearGaussianProbabilityPath,
    SchrodingerBridgeProbabilityPath,
    VariancePreservingDiracProbabilityPath,
)
from sc_flow.backends.torch.probability_paths._utils import (
    get_probability_path,
    verify_probability_path_dictionary,
)

batch_size = 8
num_feats = 16
num_channels = 3
height = 32
width = 64


invalid_key = "invalid_key"
non_deterministic_probability_path_ids = [
    "constant-noise-linear-gaussian",
    "cnlg-pp",
    "schrodinger-bridge-gaussian",
    "sbg-pp",
]


compute_mu_t_kwargs = {}
compute_ut_kwargs = {}
compute_sigma_t_kwargs = {}


def valid_compute_mu_t_fn(
    t: torch.Tensor,
    x0: torch.Tensor,
    x1: torch.Tensor,
) -> torch.Tensor:
    return (1 - t) * x1 - t * x0


def valid_compute_ut_fn(
    t: torch.Tensor,
    xt: torch.Tensor,
    x0: torch.Tensor,
    x1: torch.Tensor,
) -> torch.Tensor:
    return x1 - x0


def valid_compute_sigma_t_fn(
    t: torch.Tensor,
) -> torch.Tensor:
    return t


def invalid_compute_mu_t_fn(
    t: torch.Tensor,
    x1: torch.Tensor,
) -> torch.Tensor:
    return (1 - t) * x1


def invalid_compute_ut_fn(
    t: torch.Tensor,
    x0: torch.Tensor,
    x1: torch.Tensor,
) -> torch.Tensor:
    return x1 - x0


def invalid_compute_sigma_t_fn(
    t: torch.Tensor,
    xt: torch.Tensor,
) -> torch.Tensor:
    return t


class TestProbabilityPathUtils:
    @pytest.mark.parametrize(
        "probability_path_id",
        [
            "constant-noise-linear-gaussian",
            "cnlg-pp",
            "schrodinger-bridge-gaussian",
            "sbg-pp",
            "variance-preserving-dirac",
            "vpd-pp",
            "linear-dirac",
            "ld-pp",
            invalid_key,
        ],
    )
    @pytest.mark.parametrize("sigma", [0.0, 1.0])
    def test_get_probability_path_with_id(
        self,
        probability_path_id: str,
        sigma: float,
    ) -> None:
        if probability_path_id == invalid_key:
            with pytest.raises(ValueError, match=r"Probability path .* not supported\."):
                probability_path = get_probability_path(probability_path_id, sigma)
            return None

        if probability_path_id in non_deterministic_probability_path_ids and sigma == 0.0:
            with pytest.raises(
                ValueError,
                match=r"Argument sigma should be a positive float",
            ):
                probability_path = get_probability_path(probability_path_id, sigma)
            return None
        probability_path = get_probability_path(probability_path_id, sigma)

        if probability_path in ["constant-noise-linear-gaussian", "cnlg-pp"]:
            assert isinstance(probability_path, LinearGaussianProbabilityPath)
        if probability_path_id in ["schrodinger-bridge", "sb-pp"]:
            assert isinstance(probability_path, SchrodingerBridgeProbabilityPath)
        if probability_path_id in ["variance-preserving", "vp-pp"]:
            assert isinstance(probability_path, VariancePreservingDiracProbabilityPath)
        if probability_path_id in ["dirac", "d-pp"]:
            assert isinstance(probability_path, LinearDiracProbabilityPath)

    @pytest.mark.parametrize(
        "probability_path_dict",
        [
            {
                MU_T_FN_KEY: valid_compute_mu_t_fn,
                U_T_FN_KEY: valid_compute_ut_fn,
                SIGMA_T_FN_KEY: valid_compute_sigma_t_fn,
            },
            {
                U_T_FN_KEY: valid_compute_ut_fn,
                SIGMA_T_FN_KEY: valid_compute_sigma_t_fn,
            },
            {
                MU_T_FN_KEY: valid_compute_mu_t_fn,
                SIGMA_T_FN_KEY: valid_compute_sigma_t_fn,
            },
            {
                MU_T_FN_KEY: valid_compute_mu_t_fn,
                U_T_FN_KEY: valid_compute_ut_fn,
            },
            {
                MU_T_FN_KEY: invalid_compute_mu_t_fn,
                U_T_FN_KEY: valid_compute_ut_fn,
                SIGMA_T_FN_KEY: valid_compute_sigma_t_fn,
            },
            {
                MU_T_FN_KEY: valid_compute_mu_t_fn,
                U_T_FN_KEY: invalid_compute_ut_fn,
                SIGMA_T_FN_KEY: valid_compute_sigma_t_fn,
            },
            {
                MU_T_FN_KEY: valid_compute_mu_t_fn,
                U_T_FN_KEY: valid_compute_ut_fn,
                SIGMA_T_FN_KEY: invalid_compute_sigma_t_fn,
            },
        ],
    )
    @pytest.mark.parametrize("sigma", [0.0, 1.0])
    @pytest.mark.parametrize("require_prng", [True, False])
    def test_get_probability_path_with_dict(
        self,
        probability_path_dict: dict[str, Callable],
        sigma: float,
        require_prng: bool,
    ) -> None:
        # missing keys (KeyError)
        if MU_T_FN_KEY not in probability_path_dict.keys():
            with pytest.raises(
                KeyError,
                match=r"",
            ):
                probability_path = get_probability_path(probability_path_dict, sigma, require_prng=require_prng)
            return None
        if U_T_FN_KEY not in probability_path_dict.keys():
            with pytest.raises(
                KeyError,
                match=r"",
            ):
                probability_path = get_probability_path(probability_path_dict, sigma, require_prng=require_prng)
            return None
        if require_prng and SIGMA_T_FN_KEY not in probability_path_dict.keys():
            with pytest.raises(
                KeyError,
                match=r"",
            ):
                probability_path = get_probability_path(probability_path_dict, sigma, require_prng=require_prng)
            return None
        # wrong function signature (TypeError)
        if ["t", "x0", "x1"] != list(get_fn_args_names_and_types(probability_path_dict[MU_T_FN_KEY]).keys()):
            with pytest.raises(
                TypeError,
                match=r"",
            ):
                probability_path = get_probability_path(probability_path_dict, sigma, require_prng=require_prng)
            return None
        if ["t", "xt", "x0", "x1"] != list(get_fn_args_names_and_types(probability_path_dict[U_T_FN_KEY]).keys()):
            print(list(get_fn_args_names_and_types(probability_path_dict[U_T_FN_KEY]).keys()))
            with pytest.raises(
                TypeError,
                match=r"",
            ):
                probability_path = get_probability_path(probability_path_dict, sigma, require_prng=require_prng)
            return None
        if require_prng and [
            "t",
        ] != list(get_fn_args_names_and_types(probability_path_dict[SIGMA_T_FN_KEY]).keys()):
            with pytest.raises(
                TypeError,
                match=r"",
            ):
                probability_path = get_probability_path(probability_path_dict, sigma, require_prng=require_prng)
            return None
        # wrong sigma value (ValueError)
        if require_prng and sigma == 0.0:
            with pytest.raises(
                ValueError,
                match=r"",
            ):
                probability_path = get_probability_path(probability_path_dict, sigma, require_prng=require_prng)
            return None
        probability_path = get_probability_path(probability_path_dict, sigma, require_prng=require_prng)
        assert isinstance(probability_path, BaseProbabilityPath)

    @pytest.mark.parametrize(
        "probability_path_dict",
        [
            {
                MU_T_FN_KEY: valid_compute_mu_t_fn,
                U_T_FN_KEY: valid_compute_ut_fn,
                SIGMA_T_FN_KEY: valid_compute_sigma_t_fn,
            },
            {
                U_T_FN_KEY: valid_compute_ut_fn,
                SIGMA_T_FN_KEY: valid_compute_sigma_t_fn,
            },
            {
                MU_T_FN_KEY: valid_compute_mu_t_fn,
                SIGMA_T_FN_KEY: valid_compute_sigma_t_fn,
            },
            {
                MU_T_FN_KEY: valid_compute_mu_t_fn,
                U_T_FN_KEY: valid_compute_ut_fn,
            },
            {
                MU_T_FN_KEY: invalid_compute_mu_t_fn,
                U_T_FN_KEY: valid_compute_ut_fn,
                SIGMA_T_FN_KEY: valid_compute_sigma_t_fn,
            },
            {
                MU_T_FN_KEY: valid_compute_mu_t_fn,
                U_T_FN_KEY: invalid_compute_ut_fn,
                SIGMA_T_FN_KEY: valid_compute_sigma_t_fn,
            },
            {
                MU_T_FN_KEY: valid_compute_mu_t_fn,
                U_T_FN_KEY: valid_compute_ut_fn,
                SIGMA_T_FN_KEY: invalid_compute_sigma_t_fn,
            },
        ],
    )
    @pytest.mark.parametrize("require_prng", [True, False])
    def test_verify_probability_path_dict(
        self,
        probability_path_dict: dict[str, Callable],
        require_prng: bool,
    ) -> None:
        # missing keys (KeyError)
        if MU_T_FN_KEY not in probability_path_dict.keys():
            with pytest.raises(
                KeyError,
                match=r"",
            ):
                verify_probability_path_dictionary(
                    probability_path_dict,
                    require_prng,
                    compute_mu_t_kwargs,
                    compute_ut_kwargs,
                    compute_sigma_t_kwargs,
                )
            return None
        if U_T_FN_KEY not in probability_path_dict.keys():
            with pytest.raises(
                KeyError,
                match=r"",
            ):
                verify_probability_path_dictionary(
                    probability_path_dict,
                    require_prng,
                    compute_mu_t_kwargs,
                    compute_ut_kwargs,
                    compute_sigma_t_kwargs,
                )
            return None
        if require_prng and SIGMA_T_FN_KEY not in probability_path_dict.keys():
            with pytest.raises(
                KeyError,
                match=r"",
            ):
                verify_probability_path_dictionary(
                    probability_path_dict,
                    require_prng,
                    compute_mu_t_kwargs,
                    compute_ut_kwargs,
                    compute_sigma_t_kwargs,
                )
            return None
        # wrong function signature (TypeError)
        if ["t", "x0", "x1"] != list(get_fn_args_names_and_types(probability_path_dict[MU_T_FN_KEY]).keys()):
            with pytest.raises(
                TypeError,
                match=r"",
            ):
                verify_probability_path_dictionary(
                    probability_path_dict,
                    require_prng,
                    compute_mu_t_kwargs,
                    compute_ut_kwargs,
                    compute_sigma_t_kwargs,
                )
            return None
        if ["t", "xt", "x0", "x1"] != list(get_fn_args_names_and_types(probability_path_dict[U_T_FN_KEY]).keys()):
            with pytest.raises(
                TypeError,
                match=r"",
            ):
                verify_probability_path_dictionary(
                    probability_path_dict,
                    require_prng,
                    compute_mu_t_kwargs,
                    compute_ut_kwargs,
                    compute_sigma_t_kwargs,
                )
            return None
        if require_prng and [
            "t",
        ] != list(get_fn_args_names_and_types(probability_path_dict[SIGMA_T_FN_KEY]).keys()):
            with pytest.raises(
                TypeError,
                match=r"",
            ):
                verify_probability_path_dictionary(
                    probability_path_dict,
                    require_prng,
                    compute_mu_t_kwargs,
                    compute_ut_kwargs,
                    compute_sigma_t_kwargs,
                )
            return None
        verify_probability_path_dictionary(
            probability_path_dict,
            require_prng,
            compute_mu_t_kwargs,
            compute_ut_kwargs,
            compute_sigma_t_kwargs,
        )

    @pytest.mark.parametrize("compute_mu_t_fn", [valid_compute_mu_t_fn, invalid_compute_mu_t_fn])
    @pytest.mark.parametrize("compute_ut_fn", [valid_compute_ut_fn, invalid_compute_ut_fn])
    @pytest.mark.parametrize("compute_sigma_t_fn", [None, valid_compute_sigma_t_fn, invalid_compute_sigma_t_fn])
    @pytest.mark.parametrize("sigma", [0.0, 1.0])
    @pytest.mark.parametrize("require_prng", [True, False])
    @pytest.mark.parametrize("method", ["compute_xt", "compute_mu_t", "compute_ut"])
    def test_custom_probability_path(
        self,
        compute_mu_t_fn: Callable,
        compute_ut_fn: Callable,
        compute_sigma_t_fn: Callable | None,
        sigma: float,
        require_prng: bool,
        method: str,
    ) -> None:
        """"""

        # initializing custom probability path
        probability_path_dict = {
            MU_T_FN_KEY: valid_compute_mu_t_fn,
            U_T_FN_KEY: valid_compute_ut_fn,
            SIGMA_T_FN_KEY: valid_compute_sigma_t_fn,
        }

        # missing keys (KeyError)
        if MU_T_FN_KEY not in probability_path_dict.keys():
            with pytest.raises(
                KeyError,
                match=r"",
            ):
                probability_path = get_probability_path(probability_path_dict, sigma, require_prng=require_prng)
            return None
        if U_T_FN_KEY not in probability_path_dict.keys():
            with pytest.raises(
                KeyError,
                match=r"",
            ):
                probability_path = get_probability_path(probability_path_dict, sigma, require_prng=require_prng)
            return None
        if require_prng and SIGMA_T_FN_KEY not in probability_path_dict.keys():
            with pytest.raises(
                KeyError,
                match=r"",
            ):
                probability_path = get_probability_path(probability_path_dict, sigma, require_prng=require_prng)
            return None
        # wrong function signature (TypeError)
        if ["t", "x0", "x1"] != list(get_fn_args_names_and_types(probability_path_dict[MU_T_FN_KEY]).keys()):
            with pytest.raises(
                TypeError,
                match=r"",
            ):
                probability_path = get_probability_path(probability_path_dict, sigma, require_prng=require_prng)
            return None
        if ["t", "xt", "x0", "x1"] != list(get_fn_args_names_and_types(probability_path_dict[U_T_FN_KEY]).keys()):
            with pytest.raises(
                TypeError,
                match=r"",
            ):
                probability_path = get_probability_path(probability_path_dict, sigma, require_prng=require_prng)
            return None
        if require_prng and [
            "t",
        ] != list(get_fn_args_names_and_types(probability_path_dict[SIGMA_T_FN_KEY]).keys()):
            with pytest.raises(
                TypeError,
                match=r"",
            ):
                probability_path = get_probability_path(probability_path_dict, sigma, require_prng=require_prng)
            return None
        # wrong sigma value (ValueError)
        if require_prng and sigma == 0.0:
            with pytest.raises(
                ValueError,
                match=r"",
            ):
                probability_path = get_probability_path(probability_path_dict, sigma, require_prng=require_prng)
            return None
        probability_path = get_probability_path(probability_path_dict, sigma, require_prng=require_prng)
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
