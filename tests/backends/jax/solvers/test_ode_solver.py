import diffrax as dfx
import flax.linen as nn
import jax
import jax.numpy as jnp
import pytest

from sc_flow.backends.jax._types import ArrayLike, TVfFn
from sc_flow.backends.jax.nn import BaseVelocityField
from sc_flow.backends.jax.solvers._ode_solver import ODESolver


class SimpleLinearVF(BaseVelocityField):
    """VF for tests: dy/dt = a * y."""

    a: float = 1.0

    def _make_vf(self) -> nn.Module:
        return self

    def __call__(self, t: ArrayLike, xt: ArrayLike, *args, **kwargs) -> ArrayLike:
        a = float(kwargs.get("a", self.a))
        return a * xt

    def get_vf_fn(self, *args, **vf_kwargs) -> TVfFn:
        a = float(vf_kwargs.get("a", self.a))

        def vf(t: ArrayLike, x: ArrayLike, _args) -> ArrayLike:
            return a * x

        return vf


def test_exp_growth_matches_analytic():
    vf = SimpleLinearVF(a=1.0)
    solver = ODESolver(dynamics=vf, vf_kwargs={"a": 1.0})

    y0 = jnp.array([1.0])
    y1 = solver.solve(y0, num_time_steps=500)

    assert jnp.allclose(y1, jnp.e, rtol=1e-2, atol=1e-2)


def test_return_trajectory_shape_and_consistency():
    vf = SimpleLinearVF()
    num_time_steps = 50
    kwargs = {"a": -0.5}

    solver = ODESolver(dynamics=vf, vf_kwargs=kwargs)
    y0 = jnp.array([1.0, 2.0])

    y_final = solver.solve(y0, num_time_steps=num_time_steps)
    traj = solver.solve(y0, num_time_steps=num_time_steps, return_trajectory=True)

    assert traj.shape == (num_time_steps,) + y0.shape
    assert jnp.allclose(traj[0], y0)
    assert jnp.allclose(traj[-1], y_final, rtol=1e-3, atol=1e-3)


def test_vf_kwargs_are_passed_through():
    vf = SimpleLinearVF(a=1.0)
    solver = ODESolver(dynamics=vf, vf_kwargs={"a": 0.0})

    y0 = jnp.array([1.0])
    traj = solver.solve(y0, num_time_steps=5, return_trajectory=True)

    assert jnp.allclose(traj, jnp.broadcast_to(y0, traj.shape))


def test_stepsize_controller_validation():
    vf = SimpleLinearVF()
    solver = ODESolver(dynamics=vf, method=dfx.Tsit5())
    y0 = jnp.array([1.0])

    controller = dfx.PIDController(rtol=1e-3, atol=1e-6)
    solver.solve(
        y0,
        num_time_steps=50,
        solver_kwargs={"stepsize_controller": controller},
    )

    with pytest.raises(TypeError):
        solver.solve(
            y0,
            num_time_steps=50,
            solver_kwargs={"stepsize_controller": object()},
        )


def test_device_object_is_respected():
    device = jax.devices()[0]
    vf = SimpleLinearVF()
    solver = ODESolver(dynamics=vf, device_id=device)

    y0 = jnp.array([1.0])
    y1 = solver.solve(y0)

    arr_device = getattr(y1, "device", None)
    if arr_device is None:
        arr_device = y1.device()

    assert arr_device == device


def test_odesolver_dynamics_and_vf_types():
    vf = SimpleLinearVF(a=2.0)
    solver = ODESolver(dynamics=vf, vf_kwargs={"a": 2.0})

    assert isinstance(solver.dynamics, BaseVelocityField)

    vf_fn = solver.vf
    assert callable(vf_fn)

    t = jnp.array(0.0)
    x = jnp.array([3.0])
    args = None

    y = vf_fn(t, x, args)
    assert isinstance(y, jnp.ndarray)
    assert y.shape == x.shape
    assert jnp.allclose(y, 2.0 * x)
