from collections.abc import Callable
from typing import Literal

import pytest

from sc_flow._constants import MU_T_FN_KEY, SIGMA_T_FN_KEY, U_T_FN_KEY
from sc_flow._runtime import (
    raise_runtime_error_on_backend_failed_import,
    raise_runtime_error_on_backend_not_supported,
    set_jax_import_failed,
    set_torch_import_failed,
)
from sc_flow._utils import get_fn_args_names_and_types
from sc_flow.backends._probability_path_utils import get_probability_path
from sc_flow.backends.jax.probability_paths._probability_paths import BaseProbabilityPath as JaxBaseProbabilityPath
from sc_flow.backends.torch.probability_paths._probability_paths import BaseProbabilityPath as TorchBaseProbabilityPath

BaseProbabilityPath = TorchBaseProbabilityPath | JaxBaseProbabilityPath


def verify_method_output(
    probability_path: BaseProbabilityPath,
    method: Literal["compute_xt", "compute_mu_t", "compute_ut"],
    batch_size: int,
    num_feats: int,
    num_channels: int,
    height: int,
    width: int,
) -> None:
    """"""
    from sc_flow._runtime import BACKEND

    if BACKEND == "torch":
        try:
            from torch import zeros
        except (ImportError, ModuleNotFoundError):
            set_torch_import_failed(True)
            raise_runtime_error_on_backend_failed_import()
    elif BACKEND == "jax":
        try:
            from jax.numpy import zeros
            from jax.random import PRNGKey
        except (ImportError, ModuleNotFoundError):
            set_jax_import_failed(True)
            raise_runtime_error_on_backend_failed_import()
    else:
        raise_runtime_error_on_backend_not_supported(BACKEND)

    tested_method = getattr(probability_path, method)

    # 2D - case 0: correct inputs
    t = zeros((batch_size, 1))
    x0 = zeros((batch_size, num_feats))
    x1 = zeros((batch_size, num_feats))

    if method == "compute_xt" and BACKEND == "jax":
        out = tested_method(t, x0, x1, prng=PRNGKey(0))
    elif method == "compute_ut":
        xt = zeros((batch_size, num_feats))
        out = tested_method(t, xt, x0, x1)
    else:
        out = tested_method(t, x0, x1)
    assert out.shape == (batch_size, num_feats)

    # 2D - case 1: shape mismatch between x0 and x1
    t = zeros((batch_size, 1))
    x0 = zeros((batch_size, num_feats))
    x1 = zeros((batch_size, num_feats + 1))
    with pytest.raises(
        ValueError,
        # match=r"`input` and `target` are supposed to have the same shape"
    ):
        if method == "compute_xt" and BACKEND == "jax":
            out = tested_method(t, x0, x1, prng=PRNGKey(0))
        elif method == "compute_ut":
            xt = zeros((batch_size, num_feats))
            out = tested_method(t, xt, x0, x1)
        else:
            out = tested_method(t, x0, x1)

    # 3D - case 0: correct inputs
    t = zeros((batch_size, 1))
    x0 = zeros((batch_size, num_channels, height, width))
    x1 = zeros((batch_size, num_channels, height, width))
    if method == "compute_xt" and BACKEND == "jax":
        out = tested_method(t, x0, x1, prng=PRNGKey(0))
    elif method == "compute_ut":
        xt = zeros((batch_size, num_channels, height, width))
        out = tested_method(t, xt, x0, x1)
    else:
        out = tested_method(t, x0, x1)
    assert out.shape == (batch_size, num_channels, height, width)

    # 3D - case 1: shape mismatch between x0 and x1
    t = zeros((batch_size, 1))
    x0 = zeros((batch_size, num_channels, height, width))
    x1 = zeros((batch_size, num_channels, height, width + 1))
    with pytest.raises(
        ValueError,
        # match=r"`input_tensor` and `target_tensor` are supposed to have the same shape"
    ):
        if method == "compute_xt" and BACKEND == "jax":
            out = tested_method(t, x0, x1, prng=PRNGKey(0))
        elif method == "compute_ut":
            xt = zeros((batch_size, num_channels, height, width))
            out = tested_method(t, xt, x0, x1)
        else:
            out = tested_method(t, x0, x1)


def verify_get_probability_path_with_dict(
    probability_path_dict: dict[str, Callable],
    sigma: float,
    require_prng: bool,
    method: str,
    batch_size: int,
    num_feats: int,
    num_channels: int,
    height: int,
    width: int,
) -> None:
    # missing keys (KeyError)
    if MU_T_FN_KEY not in probability_path_dict.keys():
        with pytest.raises(
            KeyError,
            match=r"",
        ):
            probability_path = get_probability_path(sigma, probability_path_dict, require_prng=require_prng)
        return None
    if U_T_FN_KEY not in probability_path_dict.keys():
        with pytest.raises(
            KeyError,
            match=r"",
        ):
            probability_path = get_probability_path(sigma, probability_path_dict, require_prng=require_prng)
        return None
    if require_prng and SIGMA_T_FN_KEY not in probability_path_dict.keys():
        with pytest.raises(
            KeyError,
            match=r"",
        ):
            probability_path = get_probability_path(sigma, probability_path_dict, require_prng=require_prng)
        return None
    # wrong function signature (TypeError)
    if ["t", "x0", "x1"] != list(get_fn_args_names_and_types(probability_path_dict[MU_T_FN_KEY]).keys()):
        with pytest.raises(
            TypeError,
            match=r"",
        ):
            probability_path = get_probability_path(sigma, probability_path_dict, require_prng=require_prng)
        return None
    if ["t", "xt", "x0", "x1"] != list(get_fn_args_names_and_types(probability_path_dict[U_T_FN_KEY]).keys()):
        print(list(get_fn_args_names_and_types(probability_path_dict[U_T_FN_KEY]).keys()))
        with pytest.raises(
            TypeError,
            match=r"",
        ):
            probability_path = get_probability_path(sigma, probability_path_dict, require_prng=require_prng)
        return None
    if require_prng and [
        "t",
    ] != list(get_fn_args_names_and_types(probability_path_dict[SIGMA_T_FN_KEY]).keys()):
        with pytest.raises(
            TypeError,
            match=r"",
        ):
            probability_path = get_probability_path(sigma, probability_path_dict, require_prng=require_prng)
        return None
    # wrong sigma value (ValueError)
    if require_prng and sigma == 0.0:
        with pytest.raises(
            ValueError,
            match=r"",
        ):
            probability_path = get_probability_path(sigma, probability_path_dict, require_prng=require_prng)
        return None
    probability_path = get_probability_path(sigma, probability_path_dict, require_prng=require_prng)
    assert isinstance(probability_path, BaseProbabilityPath)
    verify_method_output(
        probability_path,
        method,
        batch_size,
        num_feats,
        num_channels,
        height,
        width,
    )


def verify_get_probability_path_with_fns(
    compute_mu_t_fn: Callable | None,
    compute_ut_fn: Callable | None,
    compute_sigma_t_fn: Callable | None,
    sigma: float,
    require_prng: bool,
    method: str,
    batch_size: int,
    num_feats: int,
    num_channels: int,
    height: int,
    width: int,
) -> None:
    # missing arguments (ValueError)
    if compute_mu_t_fn is None:
        with pytest.raises(ValueError, match=r""):
            probability_path = get_probability_path(
                sigma,
                compute_mu_t_fn=compute_mu_t_fn,
                compute_ut_fn=compute_ut_fn,
                compute_sigma_t_fn=compute_sigma_t_fn,
                require_prng=require_prng,
            )
        return None
    if compute_ut_fn is None:
        with pytest.raises(ValueError, match=r""):
            probability_path = get_probability_path(
                sigma,
                compute_mu_t_fn=compute_mu_t_fn,
                compute_ut_fn=compute_ut_fn,
                compute_sigma_t_fn=compute_sigma_t_fn,
                require_prng=require_prng,
            )
        return None
    if compute_sigma_t_fn is None and require_prng:
        with pytest.raises(ValueError, match=r""):
            probability_path = get_probability_path(
                sigma,
                compute_mu_t_fn=compute_mu_t_fn,
                compute_ut_fn=compute_ut_fn,
                compute_sigma_t_fn=compute_sigma_t_fn,
                require_prng=require_prng,
            )
        return None
    # wrong function signature (TypeError)
    if ["t", "x0", "x1"] != list(get_fn_args_names_and_types(compute_mu_t_fn).keys()):
        with pytest.raises(
            TypeError,
            match=r"",
        ):
            probability_path = get_probability_path(
                sigma,
                compute_mu_t_fn=compute_mu_t_fn,
                compute_ut_fn=compute_ut_fn,
                compute_sigma_t_fn=compute_sigma_t_fn,
                require_prng=require_prng,
            )
        return None
    if ["t", "xt", "x0", "x1"] != list(get_fn_args_names_and_types(compute_ut_fn).keys()):
        with pytest.raises(
            TypeError,
            match=r"",
        ):
            probability_path = get_probability_path(
                sigma,
                compute_mu_t_fn=compute_mu_t_fn,
                compute_ut_fn=compute_ut_fn,
                compute_sigma_t_fn=compute_sigma_t_fn,
                require_prng=require_prng,
            )
        return None
    if require_prng and [
        "t",
    ] != list(get_fn_args_names_and_types(compute_sigma_t_fn).keys()):
        with pytest.raises(
            TypeError,
            match=r"",
        ):
            probability_path = get_probability_path(
                sigma,
                compute_mu_t_fn=compute_mu_t_fn,
                compute_ut_fn=compute_ut_fn,
                compute_sigma_t_fn=compute_sigma_t_fn,
                require_prng=require_prng,
            )
        return None
    # wrong sigma value (ValueError)
    if sigma == 0.0 and require_prng:
        with pytest.raises(ValueError, match=r""):
            probability_path = get_probability_path(
                sigma,
                compute_mu_t_fn=compute_mu_t_fn,
                compute_ut_fn=compute_ut_fn,
                compute_sigma_t_fn=compute_sigma_t_fn,
                require_prng=require_prng,
            )
        return None
    probability_path = get_probability_path(
        sigma,
        compute_mu_t_fn=compute_mu_t_fn,
        compute_ut_fn=compute_ut_fn,
        compute_sigma_t_fn=compute_sigma_t_fn,
        require_prng=require_prng,
    )
    assert isinstance(probability_path, BaseProbabilityPath)
    verify_method_output(
        probability_path,
        method,
        batch_size,
        num_feats,
        num_channels,
        height,
        width,
    )
