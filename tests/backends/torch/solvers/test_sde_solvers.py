from dataclasses import dataclass

import diffrax as dfx
import jax
import numpy as np
import pytest
import torch
from torch import Tensor, nn

from sc_flow.backends.torch.solvers.wrapperjax.sde_solver import WrappedSDESolver

# dataclassics ───────────────────────────────────────────────────────────────────


@dataclass
class ConstantDriftVF:
    a: nn.Parameter

    def get_vf_fn(self, **vf_kwargs):
        a = self.a

        def f(t: Tensor, y: Tensor) -> Tensor:
            return a * torch.ones_like(y)

        return f

    def zero_grad(self):
        if self.a.grad is not None:
            self.a.grad.zero_()


class LinearDriftModule(nn.Module):
    def __init__(self, dim: int = 3):
        super().__init__()
        self.W = nn.Parameter(torch.eye(dim) * 0.1)
        self.b = nn.Parameter(torch.zeros(dim))

    def get_vf_fn(self, **vf_kwargs):
        W = self.W
        b = self.b

        def f(t: Tensor, y: Tensor) -> Tensor:
            return y @ W.T + b

        return f

    def zero_grad(self):
        for p in self.parameters():
            if p.grad is not None:
                p.grad.zero_()


def constant_diffusion_scalar(sigma: float = 0.1):
    def diffusion(t: Tensor, y: Tensor) -> Tensor:
        return torch.full_like(y, sigma)

    return diffusion


def time_varying_diffusion(t: Tensor, y: Tensor) -> Tensor:
    return torch.full_like(y, 0.1 * (1.0 + t))


# ── fixtures ───────────────────────────────────────────────────────────────────

DIM = 3
T0, T1 = 0.0, 1.0
NUM_STEPS = 51


@pytest.fixture
def constant_drift():
    return ConstantDriftVF(a=nn.Parameter(torch.tensor(0.3)))


@pytest.fixture
def linear_drift_module():
    return LinearDriftModule(dim=DIM)


@pytest.fixture
def y_init():
    return torch.tensor([1.0, 0.0, -2.0])


@pytest.fixture
def brownian_motion():
    return dfx.VirtualBrownianTree(
        t0=T0,
        t1=T1,
        shape=(DIM,),
        tol=1e-3,
        key=jax.random.PRNGKey(0),
    )


# ── forward tests ──────────────────────────────────────────────────────────────


def test_sde_solve_returns_correct_shape_final_state(constant_drift, y_init, brownian_motion):
    solver = WrappedSDESolver(
        dynamics=(constant_drift, constant_diffusion_scalar(0.1)),
        method="euler",
        device_id="cpu",
    )
    out = solver.solve(
        source=y_init, t0=T0, t1=T1, brownian_motion=brownian_motion, num_time_steps=NUM_STEPS, return_trajectory=False
    )
    assert out.shape == y_init.shape


def test_sde_solve_returns_correct_shape_trajectory(constant_drift, y_init, brownian_motion):
    solver = WrappedSDESolver(
        dynamics=(constant_drift, constant_diffusion_scalar(0.1)),
        method="euler",
        device_id="cpu",
    )
    out = solver.solve(
        source=y_init, t0=T0, t1=T1, brownian_motion=brownian_motion, num_time_steps=NUM_STEPS, return_trajectory=True
    )
    assert out.shape[0] == NUM_STEPS
    assert out.shape[1:] == y_init.shape


def test_sde_solve_with_zero_diffusion_matches_ode(constant_drift, y_init):
    bm = dfx.VirtualBrownianTree(t0=T0, t1=T1, shape=(DIM,), tol=1e-3, key=jax.random.PRNGKey(0))
    solver = WrappedSDESolver(
        dynamics=(constant_drift, constant_diffusion_scalar(0.0)),
        method="euler",
        device_id="cpu",
    )
    final = solver.solve(source=y_init, t0=T0, t1=T1, brownian_motion=bm, num_time_steps=501, return_trajectory=False)
    expected = y_init + 0.3 * (T1 - T0)
    final_cpu = torch.tensor(np.array(final)) if not isinstance(final, torch.Tensor) else final.to("cpu")
    assert torch.allclose(final_cpu.float(), expected.float(), atol=1e-2)


def test_sde_solve_nn_module_drift_forward(linear_drift_module, y_init, brownian_motion):
    solver = WrappedSDESolver(
        dynamics=(linear_drift_module, constant_diffusion_scalar(0.05)),
        method="euler",
        device_id="cpu",
    )
    out = solver.solve(
        source=y_init, t0=T0, t1=T1, brownian_motion=brownian_motion, num_time_steps=NUM_STEPS, return_trajectory=False
    )
    assert out.shape == y_init.shape


def test_sde_solve_time_varying_diffusion(constant_drift, y_init, brownian_motion):
    solver = WrappedSDESolver(
        dynamics=(constant_drift, time_varying_diffusion),
        method="euler",
        device_id="cpu",
    )
    out = solver.solve(
        source=y_init, t0=T0, t1=T1, brownian_motion=brownian_motion, num_time_steps=NUM_STEPS, return_trajectory=False
    )
    assert out.shape == y_init.shape


def test_sde_different_seeds_give_different_results(constant_drift, y_init):
    solver = WrappedSDESolver(
        dynamics=(constant_drift, constant_diffusion_scalar(0.5)),
        method="euler",
        device_id="cpu",
    )
    bm1 = dfx.VirtualBrownianTree(t0=T0, t1=T1, shape=(DIM,), tol=1e-3, key=jax.random.PRNGKey(0))
    bm2 = dfx.VirtualBrownianTree(t0=T0, t1=T1, shape=(DIM,), tol=1e-3, key=jax.random.PRNGKey(42))
    out1 = solver.solve(
        source=y_init, t0=T0, t1=T1, brownian_motion=bm1, num_time_steps=NUM_STEPS, return_trajectory=False
    )
    out2 = solver.solve(
        source=y_init, t0=T0, t1=T1, brownian_motion=bm2, num_time_steps=NUM_STEPS, return_trajectory=False
    )
    assert not torch.allclose(torch.tensor(np.array(out1)), torch.tensor(np.array(out2)))


# ── backward tests ─────────────────────────────────────────────────────────────


def test_sde_gradients_constant_drift_dataclass(constant_drift, y_init):
    bm = dfx.VirtualBrownianTree(t0=T0, t1=T1, shape=(DIM,), tol=1e-3, key=jax.random.PRNGKey(0))
    solver = WrappedSDESolver(
        dynamics=(constant_drift, constant_diffusion_scalar(0.1)),
        method="euler",
        device_id="cpu",
    )
    constant_drift.zero_grad()
    final = solver.solve(
        source=y_init, t0=T0, t1=T1, brownian_motion=bm, num_time_steps=NUM_STEPS, return_trajectory=False
    )
    final.sum().backward()
    assert constant_drift.a.grad is not None, "grad did not reach a"
    assert not torch.all(constant_drift.a.grad == 0), "a.grad is all zeros"


def test_sde_gradients_nn_module_drift(linear_drift_module, y_init):
    bm = dfx.VirtualBrownianTree(t0=T0, t1=T1, shape=(DIM,), tol=1e-3, key=jax.random.PRNGKey(0))
    solver = WrappedSDESolver(
        dynamics=(linear_drift_module, constant_diffusion_scalar(0.1)),
        method="euler",
        device_id="cpu",
    )
    linear_drift_module.zero_grad()
    final = solver.solve(
        source=y_init, t0=T0, t1=T1, brownian_motion=bm, num_time_steps=NUM_STEPS, return_trajectory=False
    )
    final.sum().backward()
    assert linear_drift_module.W.grad is not None, "grad did not reach W"
    assert linear_drift_module.b.grad is not None, "grad did not reach b"
    assert not torch.all(linear_drift_module.W.grad == 0), "W.grad is all zeros"
    assert not torch.all(linear_drift_module.b.grad == 0), "b.grad is all zeros"


def test_sde_gradients_source(constant_drift, y_init):
    bm = dfx.VirtualBrownianTree(t0=T0, t1=T1, shape=(DIM,), tol=1e-3, key=jax.random.PRNGKey(0))
    solver = WrappedSDESolver(
        dynamics=(constant_drift, constant_diffusion_scalar(0.1)),
        method="euler",
        device_id="cpu",
    )
    y = y_init.clone().requires_grad_(True)
    final = solver.solve(source=y, t0=T0, t1=T1, brownian_motion=bm, num_time_steps=NUM_STEPS, return_trajectory=False)
    final.sum().backward()
    assert y.grad is not None, "grad did not reach source"


def test_sde_grad_accumulates_correctly(constant_drift, y_init):
    bm = dfx.VirtualBrownianTree(t0=T0, t1=T1, shape=(DIM,), tol=1e-3, key=jax.random.PRNGKey(0))
    solver = WrappedSDESolver(
        dynamics=(constant_drift, constant_diffusion_scalar(0.1)),
        method="euler",
        device_id="cpu",
    )
    grads = []
    for _ in range(2):
        constant_drift.zero_grad()
        final = solver.solve(
            source=y_init, t0=T0, t1=T1, brownian_motion=bm, num_time_steps=NUM_STEPS, return_trajectory=False
        )
        final.sum().backward()
        grads.append(constant_drift.a.grad.clone())
    assert torch.allclose(
        torch.tensor(np.array(grads[0])),
        torch.tensor(np.array(grads[1])),
        atol=1e-5,
    ), "Repeated backward passes should give identical grads"


def test_sde_no_grad_context_skips_grad_path(constant_drift, y_init, brownian_motion):
    solver = WrappedSDESolver(
        dynamics=(constant_drift, constant_diffusion_scalar(0.1)),
        method="euler",
        device_id="cpu",
    )
    with torch.no_grad():
        out = solver.solve(
            source=y_init,
            t0=T0,
            t1=T1,
            brownian_motion=brownian_motion,
            num_time_steps=NUM_STEPS,
            return_trajectory=False,
        )
    assert out is not None
    assert out.shape == y_init.shape
