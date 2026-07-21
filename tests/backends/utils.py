from typing import TYPE_CHECKING, Literal, Union

import pytest

from sc_flow._runtime import (
    BACKEND,
    raise_runtime_error_on_backend_failed_import,
    raise_runtime_error_on_backend_not_supported,
    set_jax_import_failed,
    set_torch_import_failed,
)

if TYPE_CHECKING:
    # Only for type checking – never imported at runtime
    from sc_flow.backends.jax.probability_paths._probability_paths import BaseProbabilityPath as JaxBaseProbabilityPath
    from sc_flow.flow.probability_paths._probability_paths import (
        BaseProbabilityPath as TorchBaseProbabilityPath,
    )

# Runtime type alias – will be resolved only when used in type annotations
BaseProbabilityPath = Union["TorchBaseProbabilityPath", "JaxBaseProbabilityPath"]


def verify_method_output(
    probability_path: BaseProbabilityPath,
    method: Literal["compute_xt", "compute_mu_t", "compute_ut"],
    batch_size: int,
    num_feats: int,
    num_channels: int,
    height: int,
    width: int,
) -> None:
    # Lazy imports: only import the backend that is actually active
    if BACKEND == "torch":
        try:
            from torch import zeros
        except ImportError:
            set_torch_import_failed(True)
            raise_runtime_error_on_backend_failed_import()
        # No need to import jax at all
        prng = None  # not used for torch
    elif BACKEND == "jax":
        try:
            from jax.numpy import zeros
            from jax.random import PRNGKey
        except ImportError:
            set_jax_import_failed(True)
            raise_runtime_error_on_backend_failed_import()
        prng = PRNGKey(0)
    else:
        raise_runtime_error_on_backend_not_supported(BACKEND)

    tested_method = getattr(probability_path, method)

    # ----- 2D tests -----
    t = zeros((batch_size, 1))
    x0 = zeros((batch_size, num_feats))
    x1 = zeros((batch_size, num_feats))

    if method == "compute_xt" and BACKEND == "jax":
        out = tested_method(t, x0, x1, prng=prng)
    elif method == "compute_ut":
        xt = zeros((batch_size, num_feats))
        out = tested_method(t, xt, x0, x1)
    else:
        out = tested_method(t, x0, x1)
    assert out.shape == (batch_size, num_feats)

    # shape mismatch
    x1_bad = zeros((batch_size, num_feats + 1))
    with pytest.raises(ValueError):
        if method == "compute_xt" and BACKEND == "jax":
            _ = tested_method(t, x0, x1_bad, prng=prng)
        elif method == "compute_ut":
            xt = zeros((batch_size, num_feats))
            _ = tested_method(t, xt, x0, x1_bad)
        else:
            _ = tested_method(t, x0, x1_bad)

    # ----- 3D tests -----
    t = zeros((batch_size, 1))
    x0 = zeros((batch_size, num_channels, height, width))
    x1 = zeros((batch_size, num_channels, height, width))

    if method == "compute_xt" and BACKEND == "jax":
        out = tested_method(t, x0, x1, prng=prng)
    elif method == "compute_ut":
        xt = zeros((batch_size, num_channels, height, width))
        out = tested_method(t, xt, x0, x1)
    else:
        out = tested_method(t, x0, x1)
    assert out.shape == (batch_size, num_channels, height, width)

    # shape mismatch
    x1_bad = zeros((batch_size, num_channels, height, width + 1))
    with pytest.raises(ValueError):
        if method == "compute_xt" and BACKEND == "jax":
            _ = tested_method(t, x0, x1_bad, prng=prng)
        elif method == "compute_ut":
            xt = zeros((batch_size, num_channels, height, width))
            _ = tested_method(t, xt, x0, x1_bad)
        else:
            _ = tested_method(t, x0, x1_bad)
