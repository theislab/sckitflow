"""End-to-end inverse-problem test.

Train a conditional velocity field to translate a source distribution by a
continuous condition, freeze it, then recover the condition that steers the
pushed-forward source to a target response by optimizing the input through the
frozen flow + potential.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

import torch

from sc_flow.backends.torch.nn._vf import MLPVelocity
from sc_flow.backends.torch.surrogate import GenerativeFlowSurrogateWrapper, SquaredErrorPotential

D = 2
COVARIATE = "drug"


class _VNet(torch.nn.Module):
    """Toy conditional velocity field ``v(t, x, c)``."""

    def __init__(self, d: int = D, cond: int = D, h: int = 64) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d + 1 + cond, h),
            torch.nn.SiLU(),
            torch.nn.Linear(h, h),
            torch.nn.SiLU(),
            torch.nn.Linear(h, d),
        )

    def forward(self, t: torch.Tensor, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        if t.ndim == 0:
            t = t.expand(x.shape[0], 1)
        return self.net(torch.cat([x, t, c], dim=-1))


def _train_translation_flow(steps: int = 2000, n: int = 256) -> _VNet:
    """Flow-match ``v(t, x, c) ~ c`` so integrating maps ``x0 -> x0 + c``."""
    torch.manual_seed(0)
    vf = _VNet()
    opt = torch.optim.Adam(vf.parameters(), lr=1e-3)
    for _ in range(steps):
        x0 = 0.1 * torch.randn(n, D)
        c = torch.randn(n, D)
        t = torch.rand(n, 1)
        xt = (1 - t) * x0 + t * (x0 + c)  # straight path, target velocity u = c
        loss = ((vf(t, xt, c) - c) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    vf.eval()
    for p in vf.parameters():
        p.requires_grad_(False)
    return vf


def test_inverse_problem_recovers_condition():
    torch.manual_seed(1)
    vf = _train_translation_flow()

    # forward model is frozen; only the candidate condition is optimized
    source = 0.1 * torch.randn(64, D)
    ystar = torch.tensor([2.0, -1.0])
    wrapper = GenerativeFlowSurrogateWrapper(vf, source, n_steps=20)
    potential = SquaredErrorPotential(ystar, wrapper, reduction_input="mean", reduction_output="none")

    c_opt = torch.zeros(1, D, requires_grad=True)
    opt = torch.optim.Adam([c_opt], lr=5e-2)

    psi0 = float(potential(c_opt).mean().detach())
    for _ in range(400):
        obj = potential(c_opt).mean()
        opt.zero_grad()
        obj.backward()
        opt.step()
    psi1 = float(potential(c_opt).mean().detach())

    # the frozen forward model was never updated
    assert all(not p.requires_grad for p in vf.parameters())
    # the inverse optimization drove the potential down by >90%
    assert psi1 < 0.1 * psi0
    # and recovered the condition that lands the population at ystar
    expected = (ystar - source.mean(0)).detach()
    assert torch.allclose(c_opt.detach().squeeze(0), expected, atol=0.1)


class _TensorConditionVF(torch.nn.Module):
    """Adapt the production ``MLPVelocity`` to the wrapper's ``(t, state, cond)`` call.

    ``MLPVelocity`` takes a set-structured ``condition_dict``; the wrapper broadcasts a
    plain condition tensor. This presents the optimized ``(K, cond_dim)`` covariate as a
    size-1 set ``{covariate: (K, 1, cond_dim)}``.
    """

    def __init__(self, vf: MLPVelocity, covariate: str = COVARIATE) -> None:
        super().__init__()
        self.vf = vf
        self.covariate = covariate

    def forward(self, t: torch.Tensor, state: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.vf(t, state, condition_dict={self.covariate: cond.unsqueeze(1)})


def _train_conditional_mlp_velocity(steps: int = 2500, n: int = 256) -> MLPVelocity:
    """Flow-match the production MLPVelocity to translate ``x0 -> x0 + c``."""
    torch.manual_seed(0)
    vf = MLPVelocity(
        state_dim=D,
        condition_encoder_input_layers={COVARIATE: {"input_dim": D, "output_dim": 16}},
        condition_encoder_output_dim=16,
        condition_encoder_pooling_proj_dim=16,
        conditioning_id="concat",
        vf_decoder_mlp_kwargs={"hidden_dims": [64, 64]},
    )
    opt = torch.optim.Adam(vf.parameters(), lr=1e-3)
    for _ in range(steps):
        x0 = 0.1 * torch.randn(n, D)
        c = torch.randn(n, D)
        t = torch.rand(n, 1)
        xt = (1 - t) * x0 + t * (x0 + c)
        loss = ((vf(t, xt, condition_dict={COVARIATE: c.unsqueeze(1)}) - c) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    vf.eval()
    for p in vf.parameters():
        p.requires_grad_(False)
    return vf


def test_inverse_with_production_velocity_field():
    """Same inverse problem, but the frozen forward model is the real MLPVelocity."""
    torch.manual_seed(1)
    vf = _train_conditional_mlp_velocity()

    source = 0.1 * torch.randn(64, D)
    ystar = torch.tensor([2.0, -1.0])
    wrapper = GenerativeFlowSurrogateWrapper(_TensorConditionVF(vf), source, n_steps=20)
    potential = SquaredErrorPotential(ystar, wrapper, reduction_input="mean", reduction_output="none")

    c_opt = torch.zeros(1, D, requires_grad=True)
    opt = torch.optim.Adam([c_opt], lr=5e-2)
    psi0 = float(potential(c_opt).mean().detach())
    for _ in range(400):
        obj = potential(c_opt).mean()
        opt.zero_grad()
        obj.backward()
        opt.step()
    psi1 = float(potential(c_opt).mean().detach())

    assert all(not p.requires_grad for p in vf.parameters())
    assert psi1 < 0.1 * psi0
    assert torch.allclose(c_opt.detach().squeeze(0), (ystar - source.mean(0)).detach(), atol=0.15)


def test_wrapper_response_shape_and_grad():
    vf = _VNet()
    source = torch.randn(8, D)
    wrapper = GenerativeFlowSurrogateWrapper(vf, source, n_steps=5)
    x = torch.randn(3, D, requires_grad=True)  # 3 candidate conditions
    y = wrapper(x)
    assert y.shape == (8, 3, D)  # (M cells, N candidates, G features)
    y.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
