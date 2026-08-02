from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import pytest

if TYPE_CHECKING:
    from sckitflow.core.probability_paths._probability_paths import BaseProbabilityPath


def get_dummy_network(input_dim, output_dim, hidden_dims=(None,), sigma=0.5):
    # set_backend(backend)

    from torch import Tensor, rand
    from torch.nn import Module
    from torch.nn.functional import mse_loss

    from sckitflow.core.nn._modules import MLP
    from sckitflow.core.probability_paths import LinearGaussianProbabilityPath

    class MethodClass(Module):
        def __init__(self, network, prob_path, time_sampler) -> None:
            super().__init__()
            self.network = network
            self.prob_path = prob_path
            self.time_sampler = time_sampler
            self.train_called = False
            self.eval_called = 0

        def train_step(self, batch, prng_step_fn=None) -> Tensor:
            target = batch["target"]
            source = batch["source"]
            batch_size = target.shape[0]
            t = self.time_sampler(batch_size, device=target.device)
            xt = self.prob_path.compute_xt(t, source, target)
            vt = self.network.forward(xt)
            ut = self.prob_path.compute_ut(t, xt, source, target)
            loss = mse_loss(vt, ut)
            self.train_called = True
            return loss

        def validation_step(self, batch, prng_step_fn=None) -> None:
            self.eval_called = 1
            pass

    network = MLP(input_dim=input_dim, output_dim=output_dim, hidden_dims=hidden_dims)
    prob_path = LinearGaussianProbabilityPath(
        sigma=sigma,
    )
    time_sampler = rand
    method = MethodClass(network=network, prob_path=prob_path, time_sampler=time_sampler)

    return method


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
