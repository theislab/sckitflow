import pytest
import torch
from torch import Tensor

from sc_flow.backends.torch.solvers import ODESolver
from tests.backends.torch.solvers.solver_test_utils import (
    DummyVelocityField,
    ConstantVelocityField,
)


@pytest.fixture
def dummy_vf() -> DummyVelocityField:
    return DummyVelocityField()


@pytest.fixture
def constant_vf() -> ConstantVelocityField:
    return ConstantVelocityField()


@pytest.fixture
def dummy_solver(dummy_vf: DummyVelocityField) -> ODESolver:
    return ODESolver(
        vf=dummy_vf,
        num_time_steps=200,
        method="euler",
        device_id="cpu",
    )


@pytest.fixture
def constant_solver(constant_vf: ConstantVelocityField) -> ODESolver:
    return ODESolver(
        vf=constant_vf,
        num_time_steps=200,
        method="euler",
        device_id="cpu",
    )


def test_solve_basic(dummy_solver: ODESolver) -> None:
    """Basic shape & type check, plus that solve calls the velocity field."""
    x_init = torch.tensor([[0.0, 0.0]])
    result = dummy_solver.solve(source=x_init)

    assert isinstance(result, Tensor)
    assert result.shape == x_init.shape


@pytest.mark.parametrize("method", ["euler", "dopri5", "midpoint", "heun3"])
def test_solve_with_different_methods(method: str, dummy_vf: DummyVelocityField) -> None:
    """
    For dx/dt = 3 t^2 on [0,1], solution is x(1) = x(0) + 1.
    Check several torchdiffeq methods give approximately that.
    """
    x_init = torch.tensor([1.0, 0.0])

    solver = ODESolver(
        vf=dummy_vf,
        num_time_steps=200,
        method=method,
        device_id="cpu",
    )
    result = solver.solve(source=x_init)

    assert isinstance(result, Tensor)
    assert torch.allclose(
        torch.tensor([2.0, 1.0]), result, atol=1e-2
    ), "The solution to dx/dt = 3 t^2 from 0 to 1 should be x0 + 1."


def test_return_trajectory(dummy_vf: DummyVelocityField) -> None:
    """return_trajectory=True should return [T, batch, dim]."""
    x_init = torch.zeros(2, 3)
    num_time_steps = 200
    solver = ODESolver(
        vf=dummy_vf,
        num_time_steps=num_time_steps,
        method="euler",
        device_id="cpu",
    )

    traj = solver.solve(source=x_init, return_trajectory=True)

    assert traj.shape == (num_time_steps, *x_init.shape)
    # Last point still matches analytic solution
    assert torch.allclose(traj[-1], x_init + 1.0, atol=1e-2)


def test_gradients_through_solver(constant_vf: ConstantVelocityField) -> None:
    """
    For dx/dt = a (constant), x(1) = x0 + a.
    With x0 = [1,0], loss = sum(x(1)) = (1 + a) + (0 + a) = 1 + 2a.
    d(loss)/d(a) = 2.
    """
    x_init = torch.tensor([1.0, 0.0])
    solver = ODESolver(
        vf=constant_vf,
        num_time_steps=100,
        method="euler",
        device_id="cpu",
    )

    constant_vf.zero_grad()
    result = solver.solve(source=x_init)
    loss = result.sum()
    loss.backward()

    assert constant_vf.a.grad is not None
    assert constant_vf.a.grad.item() == pytest.approx(2.0, rel=1e-3)


def test_vf_kwargs_are_forwarded(dummy_vf: DummyVelocityField) -> None:
    """
    Extra keyword arguments passed to solve(...) must be forwarded
    into vf.get_vf_fn(...) and then to forward(...).
    """
    x_init = torch.zeros(1, 2)
    solver = ODESolver(
        vf=dummy_vf,
        num_time_steps=10,
        method="euler",
        device_id="cpu",
    )

    cond = {"foo": torch.ones(1, 2)}
    alpha = 0.42

    _ = solver.solve(
        source=x_init,
        condition_dict=cond,
        alpha=alpha,
    )

    assert dummy_vf.last_forward_kwargs is not None
    assert "condition_dict" in dummy_vf.last_forward_kwargs
    assert "alpha" in dummy_vf.last_forward_kwargs
    assert dummy_vf.last_forward_kwargs["alpha"] == pytest.approx(alpha)


def test_ode_options_are_passed_to_odeint(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verify that the `options` dict in solve(...) is passed to torchdiffeq.odeint.
    """
    from sc_flow.backends.torch.solvers import ode_solver as ode_solver_module

    vf = DummyVelocityField()
    solver = ODESolver(
        vf=vf,
        num_time_steps=5,
        method="euler",
        device_id="cpu",
    )

    x_init = torch.zeros(1, 1)
    options = {"step_size": 1e-2}
    captured = {}

    def fake_odeint(func, y0, t, **kwargs):
        nonlocal captured
        captured = kwargs
        # Return a trivial 2-step "trajectory"
        return torch.stack([y0, y0 + 1.0])

    monkeypatch.setattr(ode_solver_module, "odeint", fake_odeint)

    _ = solver.solve(source=x_init, options=options)

    assert "options" in captured
    assert "step_size" in captured["options"]
    assert captured["options"]["step_size"] == pytest.approx(1e-2)
    assert captured["method"] == "euler"
