from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
import torch
from torch import Tensor

from sc_flow.backends.torch.solvers.wrapperjax._controllers import (
    ClipStepSizeController,
    ConstantStepSizeController,
    PIDController,
    StepTo,
)
from sc_flow.backends.torch.solvers.wrapperjax._ode_solver import WrappedODESolver as ODESolver


class PolyTimeVF:
    def __init__(self) -> None:
        self.last_get_vf_fn_kwargs: dict[str, Any] | None = None

    def get_vf_fn(self, **vf_kwargs: Any):
        self.last_get_vf_fn_kwargs = dict(vf_kwargs)

        def f(t: Tensor, y: Tensor) -> Tensor:
            return 3.0 * (t**2) * torch.ones_like(y)

        return f


@dataclass
class ConstantVF:
    a: torch.nn.Parameter

    def get_vf_fn(self, **vf_kwargs: Any):
        a = self.a

        def f(t: Tensor, y: Tensor) -> Tensor:
            return a * torch.ones_like(y)

        return f

    def zero_grad(self) -> None:
        if self.a.grad is not None:
            self.a.grad.zero_()


@pytest.fixture
def poly_vf() -> PolyTimeVF:
    return PolyTimeVF()


@pytest.fixture
def constant_vf() -> ConstantVF:
    return ConstantVF(a=torch.nn.Parameter(torch.tensor(0.3, device="cpu")))


@pytest.fixture
def y0() -> Tensor:
    return torch.tensor([[0.0, 1.0], [2.0, -3.0]], device="cpu")


@pytest.fixture
def t01() -> Tensor:
    return torch.linspace(0.0, 1.0, 101, device="cpu")


def test_solve_returns_trajectory_shape_dtype(poly_vf: PolyTimeVF, y0: Tensor, t01: Tensor) -> None:
    solver = ODESolver(dynamics=poly_vf, method="euler", device_id="cpu")

    traj = solver.solve(
        source=y0,
        t0=float(t01[0].item()),
        t1=float(t01[-1].item()),
        num_time_steps=t01.numel(),
        solver_kwargs={},
        return_trajectory=True,
    )

    assert isinstance(traj, Tensor)
    assert traj.shape == (t01.numel(), *y0.shape)
    assert traj.dtype == y0.dtype
    assert traj.device.type == "cpu"


def test_vf_kwargs_forwarded(poly_vf: PolyTimeVF, y0: Tensor, t01: Tensor) -> None:
    vf_kwargs = {"condition_dict": {"foo": torch.ones(1, 2, device="cpu")}, "alpha": 0.42}
    solver = ODESolver(dynamics=poly_vf, method="euler", device_id="cpu", vf_kwargs=vf_kwargs)

    _ = solver.solve(
        source=y0,
        t0=float(t01[0].item()),
        t1=float(t01[-1].item()),
        num_time_steps=t01.numel(),
        solver_kwargs={},
    )

    assert poly_vf.last_get_vf_fn_kwargs is not None
    assert "condition_dict" in poly_vf.last_get_vf_fn_kwargs
    assert poly_vf.last_get_vf_fn_kwargs["alpha"] == pytest.approx(0.42)


_METHODS = [
    "dopri8",
    "dopri5",
    "bosh3",
    "fehlberg2",
    "adaptive_heun",
    "euler",
    "midpoint",
    "heun2",
    "heun3",
    "rk4",
    "explicit_adams",
    "fixed_adams",
]

_SUPPORTED_METHODS = [m for m in _METHODS if m != "scipy_solver"]


@pytest.mark.parametrize("method", _SUPPORTED_METHODS)
def test_many_integrators_match_analytic_solution(method: str) -> None:
    vf = PolyTimeVF()
    y_init = torch.tensor([1.0, 0.0, -2.0], device="cpu")

    solver = ODESolver(dynamics=vf, method=method, device_id="cpu")

    solver_kwargs: dict[str, Any] = {}
    if method in {"implicit_adams"}:
        solver_kwargs["stepsize_controller"] = PIDController(rtol=1e-6, atol=1e-8)
        solver_kwargs["dt0"] = 0.05

    traj = solver.solve(
        source=y_init,
        t0=0.0,
        t1=1.0,
        num_time_steps=401,
        solver_kwargs=solver_kwargs,
        return_trajectory=True,
    )

    y1 = traj[-1]

    if method == "euler":
        atol = 3e-2
    elif method in {"rk4", "explicit_adams", "implicit_adams", "fixed_adams", "fehlberg2"}:
        atol = 2e-2
    else:
        atol = 1e-2

    assert torch.allclose(y1, y_init + 1.0, atol=atol, rtol=0.0)


def test_time_tensor_respected_uniform_grid(poly_vf: PolyTimeVF) -> None:
    y_init = torch.tensor([0.5, -1.5], device="cpu")

    solver = ODESolver(dynamics=poly_vf, method="dopri5", device_id="cpu")
    traj = solver.solve(
        source=y_init,
        t0=0.0,
        t1=1.0,
        num_time_steps=17,
        solver_kwargs={},
        return_trajectory=True,
    )

    assert traj.shape == (17, *y_init.shape)
    assert torch.allclose(traj[-1], y_init + 1.0, atol=1e-2, rtol=0.0)


def test_solver_kwargs_forwarded_to_diffrax_dt0(poly_vf: PolyTimeVF) -> None:
    y_init = torch.zeros(1, device="cpu")

    solver = ODESolver(dynamics=poly_vf, method="euler", device_id="cpu")

    traj_default = solver.solve(
        source=y_init,
        t0=0.0,
        t1=1.0,
        num_time_steps=11,
        solver_kwargs={},
        return_trajectory=True,
    )
    traj_small_dt = solver.solve(
        source=y_init,
        t0=0.0,
        t1=1.0,
        num_time_steps=11,
        solver_kwargs={"dt0": 0.01},
        return_trajectory=True,
    )

    err_default = (traj_default[-1] - (y_init + 1.0)).abs().item()
    err_small = (traj_small_dt[-1] - (y_init + 1.0)).abs().item()

    assert err_small < err_default
    assert not torch.allclose(traj_default[-1], traj_small_dt[-1], atol=1e-6, rtol=0.0)


def test_runs_on_cpu(poly_vf: PolyTimeVF, y0: Tensor, t01: Tensor) -> None:
    solver = ODESolver(dynamics=poly_vf, method="euler", device_id="cpu")
    out = solver.solve(
        source=y0,
        t0=float(t01[0].item()),
        t1=float(t01[-1].item()),
        num_time_steps=t01.numel(),
        solver_kwargs={},
    )
    assert isinstance(out, Tensor)
    assert out.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_runs_on_cuda_when_available(poly_vf: PolyTimeVF, y0: Tensor, t01: Tensor) -> None:
    solver = ODESolver(dynamics=poly_vf, method="euler", device_id="cuda")
    out = solver.solve(
        source=y0.to("cuda"),
        t0=float(t01[0].item()),
        t1=float(t01[-1].item()),
        num_time_steps=t01.numel(),
        solver_kwargs={},
        return_trajectory=False,
    )
    assert out.device.type in ("cpu", "cuda")


def test_gradients_through_solver_constant_vf(constant_vf):
    y_init = torch.tensor([1.0, 0.0, -2.0], device="cpu")
    solver = ODESolver(dynamics=constant_vf, method="euler", device_id="cpu")
    constant_vf.zero_grad()

    final = solver.solve(
        source=y_init,
        t0=0.0,
        t1=1.0,
        num_time_steps=201,
        solver_kwargs={},
        return_trajectory=False,
    )
    loss = final.sum()
    loss.backward()

    assert constant_vf.a.grad is not None


def test_stepsize_controller_constant_pass_through(poly_vf: PolyTimeVF) -> None:
    y0 = torch.tensor([0.25, -1.25, 2.0], device="cpu")

    solver = ODESolver(dynamics=poly_vf, method="dopri5", device_id="cpu")

    out = solver.solve(
        source=y0,
        t0=0.0,
        t1=1.0,
        num_time_steps=201,
        return_trajectory=True,
        solver_kwargs={
            "stepsize_controller": ConstantStepSizeController(),
            "dt0": 0.01,
        },
    )

    assert isinstance(out, Tensor)
    assert out.device.type == "cpu"
    assert out.shape == (201, *y0.shape)
    assert torch.allclose(out[-1], y0 + 1.0, atol=2e-2, rtol=0.0)


def test_stepsize_controller_pid_pass_through(poly_vf: PolyTimeVF) -> None:
    y0 = torch.tensor([1.0, 0.0, -2.0], device="cpu")

    solver = ODESolver(dynamics=poly_vf, method="dopri5", device_id="cpu")

    out = solver.solve(
        source=y0,
        t0=0.0,
        t1=1.0,
        num_time_steps=301,
        return_trajectory=True,
        solver_kwargs={
            "stepsize_controller": PIDController(rtol=1e-6, atol=1e-8),
            "dt0": 0.05,
        },
    )

    assert isinstance(out, Tensor)
    assert out.device.type == "cpu"
    assert out.shape == (301, *y0.shape)
    assert torch.allclose(out[-1], y0 + 1.0, atol=1e-2, rtol=0.0)


def test_stepsize_controller_step_to_pass_through(poly_vf: PolyTimeVF) -> None:
    y0 = torch.tensor([0.5, -1.5], device="cpu")

    ts = np.linspace(0.0, 1.0, 21).astype(np.float32)

    solver = ODESolver(dynamics=poly_vf, method="dopri5", device_id="cpu")

    out = solver.solve(
        source=y0,
        t0=0.0,
        t1=1.0,
        num_time_steps=101,
        return_trajectory=True,
        solver_kwargs={
            "stepsize_controller": StepTo(ts=ts),
        },
    )

    assert isinstance(out, Tensor)
    assert out.device.type == "cpu"
    assert out.shape == (101, *y0.shape)
    assert torch.allclose(out[-1], y0 + 1.0, atol=2e-2, rtol=0.0)


def test_stepsize_controller_clip_pass_through(poly_vf: PolyTimeVF) -> None:
    y0 = torch.tensor([0.0], device="cpu")

    inner = PIDController(rtol=1e-6, atol=1e-8)
    step_ts = np.linspace(0.0, 1.0, 6).astype(np.float32)
    jump_ts = np.array([0.5], dtype=np.float32)

    solver = ODESolver(dynamics=poly_vf, method="dopri5", device_id="cpu")

    out = solver.solve(
        source=y0,
        t0=0.0,
        t1=1.0,
        num_time_steps=151,
        return_trajectory=True,
        solver_kwargs={
            "stepsize_controller": ClipStepSizeController(
                controller=inner,
                step_ts=step_ts,
                jump_ts=jump_ts,
                store_rejected_steps=8,
            ),
            "dt0": 0.05,
        },
    )

    assert isinstance(out, Tensor)
    assert out.device.type == "cpu"
    assert out.shape == (151, *y0.shape)
    assert torch.allclose(out[-1], y0 + 1.0, atol=2e-2, rtol=0.0)


class LinearVFModule(torch.nn.Module):
    """Simple nn.Module velocity field: f(t, y) = W @ y + b"""

    def __init__(self, dim: int = 3):
        super().__init__()
        self.W = torch.nn.Parameter(torch.eye(dim) * 0.1)
        self.b = torch.nn.Parameter(torch.zeros(dim))

    def get_vf_fn(self, **vf_kwargs):
        W = self.W
        b = self.b

        def f(t, y):
            return y @ W.T + b

        return f

    def zero_grad(self):
        for p in self.parameters():
            if p.grad is not None:
                p.grad.zero_()


@pytest.fixture
def linear_vf_module() -> LinearVFModule:
    return LinearVFModule(dim=3)


def test_gradients_through_solver_nn_module(linear_vf_module: LinearVFModule) -> None:
    """
    Gradient test for nn.Module dynamics.

    ODE: dy/dt = W @ y + b
    """
    y_init = torch.tensor([1.0, 0.0, -2.0], device="cpu")

    solver = ODESolver(dynamics=linear_vf_module, method="euler", device_id="cpu")
    linear_vf_module.zero_grad()

    final = solver.solve(
        source=y_init,
        t0=0.0,
        t1=1.0,
        num_time_steps=201,
        solver_kwargs={},
        return_trajectory=False,
    )

    loss = final.sum()
    loss.backward()

    assert linear_vf_module.W.grad is not None, "grad did not reach W"
    assert linear_vf_module.b.grad is not None, "grad did not reach b"
    assert not torch.all(linear_vf_module.W.grad == 0), "W.grad is all zeros"
    assert not torch.all(linear_vf_module.b.grad == 0), "b.grad is all zeros"


def test_grad_branch_returns_correct_device(poly_vf):
    y0 = torch.tensor([1.0, 0.0, -2.0], requires_grad=True)
    solver = ODESolver(dynamics=poly_vf, method="euler", device_id="cpu")

    out = solver.solve(source=y0, t0=0.0, t1=1.0, num_time_steps=100, return_trajectory=False)

    # Check device before backward
    print("device before backward:", out.device.type)
    # assert out.device.type == "cpu", f"Expected cpu, got {out.device.type}"

    # Check grad flows and device is still correct after backward
    out.sum().backward()
    print("device after backward:", out.device.type)
    assert out.device.type == "cpu", f"Expected cpu after backward, got {out.device.type}"
    assert y0.grad is not None, "grad did not reach source"
    print("y0.grad device:", y0.grad.device.type)
    assert y0.grad.device.type == "cpu", f"Expected cpu grad, got {y0.grad.device.type}"
