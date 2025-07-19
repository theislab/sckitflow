from collections.abc import Callable

import pytest

from sc_flow._constants import MU_T_FN_KEY, SIGMA_T_FN_KEY, U_T_FN_KEY
from sc_flow._runtime import set_backend, set_jax_import_failed, set_torch_import_failed

try:
    import torch

    from sc_flow.backends.torch.probability_paths._probability_paths import (
        BaseProbabilityPath as TorchBaseProbabilityPath,
    )
    from sc_flow.backends.torch.probability_paths._probability_paths import (
        LinearDiracProbabilityPath as TorchLinearDiracProbabilityPath,
    )
    from sc_flow.backends.torch.probability_paths._probability_paths import (
        LinearGaussianProbabilityPath as TorchLinearGaussianProbabilityPath,
    )
    from sc_flow.backends.torch.probability_paths._probability_paths import (
        SchrodingerBridgeProbabilityPath as TorchSchrodingerBridgeProbabilityPath,
    )
    from sc_flow.backends.torch.probability_paths._probability_paths import (
        VariancePreservingDiracProbabilityPath as TorchVariancePreservingDiracProbabilityPath,
    )
except (ImportError, ModuleNotFoundError):
    set_torch_import_failed(True)

try:
    from jax import Array as JaxArray

    from sc_flow.backends.jax.probability_paths._probability_paths import (
        BaseProbabilityPath as JaxBaseProbabilityPath,
    )
    from sc_flow.backends.jax.probability_paths._probability_paths import (
        LinearDiracProbabilityPath as JaxLinearDiracProbabilityPath,
    )
    from sc_flow.backends.jax.probability_paths._probability_paths import (
        LinearGaussianProbabilityPath as JaxLinearGaussianProbabilityPath,
    )
    from sc_flow.backends.jax.probability_paths._probability_paths import (
        SchrodingerBridgeProbabilityPath as JaxSchrodingerBridgeProbabilityPath,
    )
    from sc_flow.backends.jax.probability_paths._probability_paths import (
        VariancePreservingDiracProbabilityPath as JaxVariancePreservingDiracProbabilityPath,
    )
except (ImportError, ModuleNotFoundError):
    set_jax_import_failed(True)


from sc_flow._runtime import JAX_IMPORT_FAILED, TORCH_IMPORT_FAILED

if (not TORCH_IMPORT_FAILED) and (not JAX_IMPORT_FAILED):
    BaseProbabilityPath = TorchBaseProbabilityPath | JaxBaseProbabilityPath
    LinearDiracProbabilityPath = TorchLinearDiracProbabilityPath | JaxLinearDiracProbabilityPath
    LinearGaussianProbabilityPath = TorchLinearGaussianProbabilityPath | JaxLinearGaussianProbabilityPath
    SchrodingerBridgeProbabilityPath = TorchSchrodingerBridgeProbabilityPath | JaxSchrodingerBridgeProbabilityPath
    VariancePreservingDiracProbabilityPath = (
        TorchVariancePreservingDiracProbabilityPath | JaxVariancePreservingDiracProbabilityPath
    )
elif (not TORCH_IMPORT_FAILED) and JAX_IMPORT_FAILED:
    BaseProbabilityPath = TorchBaseProbabilityPath
    LinearDiracProbabilityPath = TorchLinearDiracProbabilityPath
    LinearGaussianProbabilityPath = TorchLinearGaussianProbabilityPath
    SchrodingerBridgeProbabilityPath = TorchSchrodingerBridgeProbabilityPath
    VariancePreservingDiracProbabilityPath = TorchVariancePreservingDiracProbabilityPath
elif TORCH_IMPORT_FAILED and (not JAX_IMPORT_FAILED):
    BaseProbabilityPath = JaxBaseProbabilityPath
    LinearDiracProbabilityPath = JaxLinearDiracProbabilityPath
    LinearGaussianProbabilityPath = JaxLinearGaussianProbabilityPath
    SchrodingerBridgeProbabilityPath = JaxSchrodingerBridgeProbabilityPath
    VariancePreservingDiracProbabilityPath = JaxVariancePreservingDiracProbabilityPath
else:
    msg = "Could not import neither `torch` nor `jax` as backend. Please import one of them before proceeding."
    raise RuntimeError

from sc_flow.backends._probability_path_utils import (
    get_probability_path,
)

from .utils import verify_get_probability_path_with_dict, verify_get_probability_path_with_fns

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


def torch_valid_compute_mu_t_fn(
    t: torch.Tensor,
    x0: torch.Tensor,
    x1: torch.Tensor,
) -> torch.Tensor:
    return (1 - t) * x1 - t * x0


def torch_valid_compute_ut_fn(
    t: torch.Tensor,
    xt: torch.Tensor,
    x0: torch.Tensor,
    x1: torch.Tensor,
) -> torch.Tensor:
    return x1 - x0


def torch_valid_compute_sigma_t_fn(
    t: torch.Tensor,
) -> torch.Tensor:
    return t


def torch_invalid_compute_mu_t_fn(
    t: torch.Tensor,
    x1: torch.Tensor,
) -> torch.Tensor:
    return (1 - t) * x1


def torch_invalid_compute_ut_fn(
    t: torch.Tensor,
    x0: torch.Tensor,
    x1: torch.Tensor,
) -> torch.Tensor:
    return x1 - x0


def torch_invalid_compute_sigma_t_fn(
    t: torch.Tensor,
    xt: torch.Tensor,
) -> torch.Tensor:
    return t


def jax_valid_compute_mu_t_fn(
    t: JaxArray,
    x0: JaxArray,
    x1: JaxArray,
) -> JaxArray:
    return (1 - t) * x1 - t * x0


def jax_valid_compute_ut_fn(
    t: JaxArray,
    xt: JaxArray,
    x0: JaxArray,
    x1: JaxArray,
) -> JaxArray:
    return x1 - x0


def jax_valid_compute_sigma_t_fn(
    t: JaxArray,
) -> JaxArray:
    return t


def jax_invalid_compute_mu_t_fn(
    t: JaxArray,
    x1: JaxArray,
) -> JaxArray:
    return (1 - t) * x1


def jax_invalid_compute_ut_fn(
    t: JaxArray,
    x0: JaxArray,
    x1: JaxArray,
) -> JaxArray:
    return x1 - x0


def jax_invalid_compute_sigma_t_fn(
    t: JaxArray,
    xt: JaxArray,
) -> JaxArray:
    return t


class TestProbabilityPathUtils:
    @pytest.mark.parametrize("backend", ["torch", "jax"])
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
        backend: str,
        probability_path_id: str,
        sigma: float,
    ) -> None:
        set_backend(backend)
        if probability_path_id == invalid_key:
            with pytest.raises(ValueError, match=r"Probability path .* not supported\."):
                probability_path = get_probability_path(sigma, probability_path_id)
            return None

        if probability_path_id in non_deterministic_probability_path_ids and sigma == 0.0:
            with pytest.raises(
                ValueError,
                match=r"",
            ):
                probability_path = get_probability_path(sigma, probability_path_id)
            return None
        probability_path = get_probability_path(sigma, probability_path_id)

        if probability_path_id in ["constant-noise-linear-gaussian", "cnlg-pp"]:
            assert isinstance(probability_path, LinearGaussianProbabilityPath)
        if probability_path_id in ["schrodinger-bridge", "sb-pp"]:
            assert isinstance(probability_path, SchrodingerBridgeProbabilityPath)
        if probability_path_id in ["variance-preserving", "vp-pp"]:
            assert isinstance(probability_path, VariancePreservingDiracProbabilityPath)
        if probability_path_id in ["dirac", "d-pp"]:
            assert isinstance(probability_path, LinearDiracProbabilityPath)

    @pytest.mark.parametrize("backend", ["torch", "jax"])
    @pytest.mark.parametrize(
        "probability_path_dict",
        [
            {
                MU_T_FN_KEY: torch_valid_compute_mu_t_fn,
                U_T_FN_KEY: torch_valid_compute_ut_fn,
                SIGMA_T_FN_KEY: torch_valid_compute_sigma_t_fn,
            },
            {
                U_T_FN_KEY: torch_valid_compute_ut_fn,
                SIGMA_T_FN_KEY: torch_valid_compute_sigma_t_fn,
            },
            {
                MU_T_FN_KEY: torch_valid_compute_mu_t_fn,
                SIGMA_T_FN_KEY: torch_valid_compute_sigma_t_fn,
            },
            {
                MU_T_FN_KEY: torch_valid_compute_mu_t_fn,
                U_T_FN_KEY: torch_valid_compute_ut_fn,
            },
            {
                MU_T_FN_KEY: torch_invalid_compute_mu_t_fn,
                U_T_FN_KEY: torch_valid_compute_ut_fn,
                SIGMA_T_FN_KEY: torch_valid_compute_sigma_t_fn,
            },
            {
                MU_T_FN_KEY: torch_valid_compute_mu_t_fn,
                U_T_FN_KEY: torch_invalid_compute_ut_fn,
                SIGMA_T_FN_KEY: torch_valid_compute_sigma_t_fn,
            },
            {
                MU_T_FN_KEY: torch_valid_compute_mu_t_fn,
                U_T_FN_KEY: torch_valid_compute_ut_fn,
                SIGMA_T_FN_KEY: torch_invalid_compute_sigma_t_fn,
            },
            {
                MU_T_FN_KEY: jax_valid_compute_mu_t_fn,
                U_T_FN_KEY: jax_valid_compute_ut_fn,
                SIGMA_T_FN_KEY: jax_valid_compute_sigma_t_fn,
            },
            {
                U_T_FN_KEY: jax_valid_compute_ut_fn,
                SIGMA_T_FN_KEY: jax_valid_compute_sigma_t_fn,
            },
            {
                MU_T_FN_KEY: jax_valid_compute_mu_t_fn,
                SIGMA_T_FN_KEY: jax_valid_compute_sigma_t_fn,
            },
            {
                MU_T_FN_KEY: jax_valid_compute_mu_t_fn,
                U_T_FN_KEY: jax_valid_compute_ut_fn,
            },
            {
                MU_T_FN_KEY: jax_invalid_compute_mu_t_fn,
                U_T_FN_KEY: jax_valid_compute_ut_fn,
                SIGMA_T_FN_KEY: jax_valid_compute_sigma_t_fn,
            },
            {
                MU_T_FN_KEY: jax_valid_compute_mu_t_fn,
                U_T_FN_KEY: jax_invalid_compute_ut_fn,
                SIGMA_T_FN_KEY: jax_valid_compute_sigma_t_fn,
            },
            {
                MU_T_FN_KEY: jax_valid_compute_mu_t_fn,
                U_T_FN_KEY: jax_valid_compute_ut_fn,
                SIGMA_T_FN_KEY: jax_invalid_compute_sigma_t_fn,
            },
        ],
    )
    @pytest.mark.parametrize("sigma", [0.0, 1.0])
    @pytest.mark.parametrize("require_prng", [True, False])
    @pytest.mark.parametrize("method", ["compute_xt", "compute_mu_t", "compute_ut"])
    def verify_get_probability_path_with_dict(
        self,
        backend: str,
        probability_path_dict: dict[str, Callable],
        sigma: float,
        require_prng: bool,
        method: str,
    ) -> None:
        set_backend(backend)
        if backend == "torch":
            if probability_path_dict[MU_T_FN_KEY].__name__.startswith("jax"):
                with pytest.raises(
                    TypeError,
                    match=r"",
                ):
                    get_probability_path(sigma, probability_path_dict, require_prng=require_prng)
                return None
            if probability_path_dict[U_T_FN_KEY].__name__.startswith("jax"):
                with pytest.raises(
                    TypeError,
                    match=r"",
                ):
                    get_probability_path(sigma, probability_path_dict, require_prng=require_prng)
                return None
            elif probability_path_dict[SIGMA_T_FN_KEY] is not None and probability_path_dict[
                SIGMA_T_FN_KEY
            ].__name__.startswith("jax"):
                with pytest.raises(
                    TypeError,
                    match=r"",
                ):
                    get_probability_path(sigma, probability_path_dict, require_prng=require_prng)
                return None
        elif backend == "jax":
            if probability_path_dict[MU_T_FN_KEY].__name__.startswith("torch"):
                with pytest.raises(
                    TypeError,
                    match=r"",
                ):
                    get_probability_path(sigma, probability_path_dict, require_prng=require_prng)
                return None
            if probability_path_dict[U_T_FN_KEY].__name__.startswith("torch"):
                with pytest.raises(
                    TypeError,
                    match=r"",
                ):
                    get_probability_path(sigma, probability_path_dict, require_prng=require_prng)
                return None

            elif probability_path_dict[SIGMA_T_FN_KEY] is not None and probability_path_dict[
                SIGMA_T_FN_KEY
            ].__name__.startswith("torch"):
                with pytest.raises(
                    TypeError,
                    match=r"",
                ):
                    get_probability_path(sigma, probability_path_dict, require_prng=require_prng)
                return None

        verify_get_probability_path_with_dict(
            probability_path_dict,
            sigma,
            require_prng,
            method,
            batch_size,
            num_feats,
            num_channels,
            height,
            width,
        )

    @pytest.mark.parametrize("backend", ["torch", "jax"])
    @pytest.mark.parametrize(
        "compute_mu_t_fn",
        [
            None,
            torch_valid_compute_mu_t_fn,
            torch_invalid_compute_mu_t_fn,
            jax_valid_compute_mu_t_fn,
            jax_invalid_compute_mu_t_fn,
        ],
    )
    @pytest.mark.parametrize(
        "compute_ut_fn",
        [
            None,
            torch_valid_compute_ut_fn,
            torch_invalid_compute_ut_fn,
            jax_valid_compute_ut_fn,
            jax_invalid_compute_ut_fn,
        ],
    )
    @pytest.mark.parametrize(
        "compute_sigma_t_fn",
        [
            None,
            torch_valid_compute_sigma_t_fn,
            torch_invalid_compute_sigma_t_fn,
            jax_valid_compute_sigma_t_fn,
            jax_invalid_compute_sigma_t_fn,
        ],
    )
    @pytest.mark.parametrize("sigma", [0.0, 1.0])
    @pytest.mark.parametrize("require_prng", [True, False])
    @pytest.mark.parametrize("method", ["compute_xt", "compute_mu_t", "compute_ut"])
    def verify_get_probability_path_with_fns(
        self,
        backend: str,
        compute_mu_t_fn: Callable | None,
        compute_ut_fn: Callable | None,
        compute_sigma_t_fn: Callable | None,
        sigma: float,
        require_prng: bool,
        method: str,
    ) -> None:
        set_backend(backend)
        if backend == "torch":
            if compute_mu_t_fn.__name__.startswith("jax"):
                get_probability_path(
                    sigma,
                    compute_mu_t_fn=compute_mu_t_fn,
                    compute_ut_fn=compute_ut_fn,
                    compute_sigma_t_fn=compute_sigma_t_fn,
                    require_prng=require_prng,
                )
            elif compute_ut_fn.__name__.startswith("jax"):
                get_probability_path(
                    sigma,
                    compute_mu_t_fn=compute_mu_t_fn,
                    compute_ut_fn=compute_ut_fn,
                    compute_sigma_t_fn=compute_sigma_t_fn,
                    require_prng=require_prng,
                )
            elif compute_sigma_t_fn is not None and compute_sigma_t_fn.__name__.startswith("jax"):
                get_probability_path(
                    sigma,
                    compute_mu_t_fn=compute_mu_t_fn,
                    compute_ut_fn=compute_ut_fn,
                    compute_sigma_t_fn=compute_sigma_t_fn,
                    require_prng=require_prng,
                )
        elif backend == "jax":
            if compute_mu_t_fn.__name__.startswith("torch"):
                get_probability_path(
                    sigma,
                    compute_mu_t_fn=compute_mu_t_fn,
                    compute_ut_fn=compute_ut_fn,
                    compute_sigma_t_fn=compute_sigma_t_fn,
                    require_prng=require_prng,
                )
            elif compute_ut_fn.__name__.startswith("torch"):
                get_probability_path(
                    sigma,
                    compute_mu_t_fn=compute_mu_t_fn,
                    compute_ut_fn=compute_ut_fn,
                    compute_sigma_t_fn=compute_sigma_t_fn,
                    require_prng=require_prng,
                )
            elif compute_sigma_t_fn is compute_sigma_t_fn.__name__.startswith("torch"):
                get_probability_path(
                    sigma,
                    compute_mu_t_fn=compute_mu_t_fn,
                    compute_ut_fn=compute_ut_fn,
                    compute_sigma_t_fn=compute_sigma_t_fn,
                    require_prng=require_prng,
                )

        verify_get_probability_path_with_fns(
            compute_mu_t_fn,
            compute_ut_fn,
            compute_sigma_t_fn,
            sigma,
            require_prng,
            method,
            batch_size,
            num_feats,
            num_channels,
            height,
            width,
        )
