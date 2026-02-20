from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
import torch
from torch import Tensor

from sc_flow.backends.torch.solvers.ode_solver import ODESolver


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
    return ConstantVF(a=torch.nn.Parameter(torch.tensor(0.3)))


@pytest.fixture
def y0() -> Tensor:
    return torch.tensor([[0.0, 1.0], [2.0, -3.0]])


@pytest.fixture
def t01() -> Tensor:
    return torch.linspace(0.0, 1.0, 101)


def test_solve_returns_trajectory_shape_dtype(poly_vf: PolyTimeVF, y0: Tensor, t01: Tensor) -> None:
    solver = ODESolver(dynamics=poly_vf, method="euler", device_id="cpu")

    traj = solver.solve(
        source=y0,
        time=t01,
        rtol=1e-6,
        atol=1e-7,
        solver_kwargs={},
        return_trajectory=True,
    )

    assert isinstance(traj, Tensor)
    assert traj.shape == (t01.numel(), *y0.shape)
    assert traj.dtype == y0.dtype


def test_vf_kwargs_forwarded(poly_vf: PolyTimeVF, y0: Tensor, t01: Tensor) -> None:
    vf_kwargs = {"condition_dict": {"foo": torch.ones(1, 2)}, "alpha": 0.42}
    solver = ODESolver(dynamics=poly_vf, method="euler", device_id="cpu", vf_kwargs=vf_kwargs)

    _ = solver.solve(
        source=y0,
        time=t01,
        rtol=1e-6,
        atol=1e-7,
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
    "implicit_adams",
    "fixed_adams",
    "scipy_solver",
]


@pytest.mark.parametrize("method", _METHODS)
def test_many_integrators_match_analytic_solution(method: str) -> None:
    vf = PolyTimeVF()
    y_init = torch.tensor([1.0, 0.0, -2.0])
    t = torch.linspace(0.0, 1.0, 401)

    solver = ODESolver(dynamics=vf, method=method, device_id="cpu")
    traj = solver.solve(
        source=y_init,
        time=t,
        rtol=1e-7,
        atol=1e-8,
        solver_kwargs={},
        return_trajectory=True,
    )

    y1 = traj[-1]
    atol = 3e-2 if method == "euler" else 1e-2
    assert torch.allclose(y1, y_init + 1.0, atol=atol, rtol=0.0)


def test_time_tensor_respected(poly_vf: PolyTimeVF) -> None:
    y_init = torch.tensor([0.5, -1.5])
    t = torch.tensor([0.0, 0.1, 0.11, 0.5, 0.9, 1.0])

    solver = ODESolver(dynamics=poly_vf, method="dopri5", device_id="cpu")
    traj = solver.solve(
        source=y_init,
        time=t,
        rtol=1e-7,
        atol=1e-8,
        solver_kwargs={},
        return_trajectory=True,
    )

    assert traj.shape == (t.numel(), *y_init.shape)
    assert torch.allclose(traj[-1], y_init + 1.0, atol=1e-4, rtol=0.0)


def test_solver_kwargs_are_forwarded_to_odeint(monkeypatch: pytest.MonkeyPatch) -> None:
    import sc_flow.backends.torch.solvers.ode_solver as ode_solver_module

    vf = PolyTimeVF()
    solver = ODESolver(dynamics=vf, method="euler", device_id="cpu")

    y_init = torch.zeros(1, 1)
    t = torch.tensor([0.0, 1.0])

    captured: dict[str, Any] = {}

    def fake_odeint(func, y0, t, **kwargs):
        captured.update(kwargs)
        return torch.stack([y0, y0 + 1.0])

    monkeypatch.setattr(ode_solver_module, "odeint", fake_odeint)

    _ = solver.solve(
        source=y_init,
        time=t,
        rtol=1e-3,
        atol=1e-5,
        solver_kwargs={"options": {"step_size": 1e-2}},
        return_trajectory=True,
    )

    assert captured["rtol"] == pytest.approx(1e-3)
    assert captured["atol"] == pytest.approx(1e-5)
    assert captured["method"] == "euler"
    assert "options" in captured
    assert captured["options"]["step_size"] == pytest.approx(1e-2)


def test_runs_on_cpu(poly_vf: PolyTimeVF, y0: Tensor, t01: Tensor) -> None:
    solver = ODESolver(dynamics=poly_vf, method="euler", device_id="cpu")
    traj = solver.solve(
        source=y0,
        time=t01,
        rtol=1e-6,
        atol=1e-7,
        solver_kwargs={},
    )
    assert traj.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_runs_on_cuda_when_available(poly_vf: PolyTimeVF, y0: Tensor, t01: Tensor) -> None:
    solver = ODESolver(dynamics=poly_vf, method="euler", device_id="cuda")
    traj = solver.solve(
        source=y0,
        time=t01,
        rtol=1e-6,
        atol=1e-7,
        solver_kwargs={},
        return_trajectory=True,
    )
    assert traj.device.type == "cuda"


def test_gradients_through_solver_constant_vf(constant_vf: ConstantVF) -> None:
    y_init = torch.tensor([1.0, 0.0, -2.0])
    t = torch.linspace(0.0, 1.0, 201)

    solver = ODESolver(dynamics=constant_vf, method="euler", device_id="cpu")

    constant_vf.zero_grad()
    traj = solver.solve(
        source=y_init,
        time=t,
        rtol=1e-7,
        atol=1e-8,
        solver_kwargs={},
        return_trajectory=True,
    )

    loss = traj[-1].sum()
    loss.backward()

    assert constant_vf.a.grad is not None
    assert float(constant_vf.a.grad) == pytest.approx(3.0, rel=1e-3, abs=1e-3)
