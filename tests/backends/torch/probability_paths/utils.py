from typing import Literal

import pytest
import torch

from sc_flow.backends.torch.probability_paths._probability_paths import BaseProbabilityPath


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
