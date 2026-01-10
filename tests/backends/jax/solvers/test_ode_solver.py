import diffrax as dfx
import flax.linen as nn
import jax
import jax.numpy as jnp
import pytest

from sc_flow.backends.jax._types import ArrayLike, TVfFn
from sc_flow.backends.jax.nn import BaseVelocityField
from sc_flow.backends.jax.solvers.ode_solver import ODESolver  # adjust path as needed


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
    solver = ODESolver(num_time_steps=500)
    vf = SimpleLinearVF()

    y0 = jnp.array([1.0])
    y1 = solver.solve(vf, y0, a=1.0)

    assert jnp.allclose(y1, jnp.e, rtol=1e-2, atol=1e-2)


def test_return_trajectory_shape_and_consistency():
    solver = ODESolver(num_time_steps=50)
    vf = SimpleLinearVF()

    y0 = jnp.array([1.0, 2.0])
    kwargs = {"a": -0.5}

    y_final = solver.solve(vf, y0, **kwargs)
    traj = solver.solve(vf, y0, return_trajectory=True, **kwargs)

    assert traj.shape == (solver.num_time_steps,) + y0.shape
    assert jnp.allclose(traj[0], y0)
    assert jnp.allclose(traj[-1], y_final, rtol=1e-3, atol=1e-3)


def test_vf_kwargs_are_passed_through():
    solver = ODESolver(num_time_steps=5)
    vf = SimpleLinearVF()

    y0 = jnp.array([1.0])

    traj = solver.solve(vf, y0, return_trajectory=True, a=0.0)

    assert jnp.allclose(traj, jnp.broadcast_to(y0, traj.shape))


def test_stepsize_controller_validation():
    solver = ODESolver(num_time_steps=50, method=dfx.Tsit5())
    vf = SimpleLinearVF()
    y0 = jnp.array([1.0])

    controller = dfx.PIDController(rtol=1e-3, atol=1e-6)
    solver.solve(
        vf,
        y0,
        solver_kwargs={"stepsize_controller": controller},
    )

    with pytest.raises(TypeError):
        solver.solve(
            vf,
            y0,
            solver_kwargs={"stepsize_controller": object()},
        )


def test_invalid_device_string_raises():
    with pytest.raises(ValueError, match="No available device found"):
        ODESolver(device_id="definitely_not_a_real_platform")


def test_device_object_is_respected():
    device = jax.devices()[0]
    solver = ODESolver(device_id=device)
    vf = SimpleLinearVF()
    y0 = jnp.array([1.0])

    y1 = solver.solve(vf, y0)

    arr_device = getattr(y1, "device", None)
    if arr_device is None:
        arr_device = y1.device()

    assert arr_device == device
