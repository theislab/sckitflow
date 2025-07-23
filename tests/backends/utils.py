from typing import Literal

import pytest

from sc_flow._runtime import (
    raise_runtime_error_on_backend_failed_import,
    raise_runtime_error_on_backend_not_supported,
    set_jax_import_failed,
    set_torch_import_failed,
)
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
