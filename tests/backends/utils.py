from typing import TYPE_CHECKING, Literal

import pytest

if TYPE_CHECKING:
    from sckitflow.backends.torch.probability_paths._probability_paths import (
        BaseProbabilityPath as TorchBaseProbabilityPath,
    )

BaseProbabilityPath = "TorchBaseProbabilityPath"


def verify_method_output(
    probability_path: BaseProbabilityPath,
    method: Literal["compute_xt", "compute_mu_t", "compute_ut"],
    batch_size: int,
    num_feats: int,
    num_channels: int,
    height: int,
    width: int,
) -> None:
    from torch import zeros

    tested_method = getattr(probability_path, method)

    # ----- 2D tests -----
    t = zeros((batch_size, 1))
    x0 = zeros((batch_size, num_feats))
    x1 = zeros((batch_size, num_feats))

    if method == "compute_ut":
        xt = zeros((batch_size, num_feats))
        out = tested_method(t, xt, x0, x1)
    else:
        out = tested_method(t, x0, x1)
    assert out.shape == (batch_size, num_feats)

    # shape mismatch
    x1_bad = zeros((batch_size, num_feats + 1))
    with pytest.raises(ValueError):
        if method == "compute_ut":
            xt = zeros((batch_size, num_feats))
            _ = tested_method(t, xt, x0, x1_bad)
        else:
            _ = tested_method(t, x0, x1_bad)

    # ----- 3D tests -----
    t = zeros((batch_size, 1))
    x0 = zeros((batch_size, num_channels, height, width))
    x1 = zeros((batch_size, num_channels, height, width))

    if method == "compute_ut":
        xt = zeros((batch_size, num_channels, height, width))
        out = tested_method(t, xt, x0, x1)
    else:
        out = tested_method(t, x0, x1)
    assert out.shape == (batch_size, num_channels, height, width)

    # shape mismatch
    x1_bad = zeros((batch_size, num_channels, height, width + 1))
    with pytest.raises(ValueError):
        if method == "compute_ut":
            xt = zeros((batch_size, num_channels, height, width))
            _ = tested_method(t, xt, x0, x1_bad)
        else:
            _ = tested_method(t, x0, x1_bad)
