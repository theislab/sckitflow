import warnings

import diffrax as dfx
import flax.linen as nn
import jax
import jax.numpy as jnp
import lineax as lx
import pytest

from sc_flow.backends.jax._types import ArrayLike, TVfFn
from sc_flow.backends.jax.nn import BaseVelocityField
from sc_flow.backends.jax.solvers._sde_solver import SDESolver
from sc_flow.backends.torch._types import TSDEDynamics


class ConstantDriftVF(BaseVelocityField):
    """Velocity field: dy/dt = value (broadcast to y's shape)."""

    value: float = 1.0

    def _make_vf(self) -> nn.Module:
        return self

    def __call__(self, t: ArrayLike, xt: ArrayLike, *args, **kwargs) -> ArrayLike:
        v = float(kwargs.get("value", self.value))
        return jnp.full_like(xt, v)

    def get_vf_fn(self, *args, **vf_kwargs) -> TVfFn:
        # The function used by SDESolver.drift_fn
        v = float(vf_kwargs.get("value", self.value))

        def vf(t: ArrayLike, x: ArrayLike, _args) -> ArrayLike:
            return jnp.full_like(x, v)

        return vf


def _diff_zero_time_state(t, y):
    return lx.DiagonalLinearOperator(jnp.zeros_like(y))


def _diff_identity_time_state(t, y):
    return lx.DiagonalLinearOperator(jnp.ones_like(y))


def _diff_identity_time_only(t, args=None):
    shape = () if args is None else jnp.shape(args)
    if shape == ():
        vec = jnp.array(1.0)
    else:
        vec = jnp.ones(shape)
    return lx.DiagonalLinearOperator(vec)


def test_sde_deterministic_zero_diffusion_matches_ode():
    """dX = 1 dt, sigma = 0  =>  X(1) = X(0) + 1."""
    drift_vf = ConstantDriftVF(value=1.0)
    dynamics: TSDEDynamics = (drift_vf, _diff_zero_time_state)

    solver = SDESolver(dynamics=dynamics, method=dfx.Euler(), device_id="cpu")
    x0 = jnp.array([0.0, 2.0, -1.0])

    ys = solver.solve(
        source=x0,
        brownian_motion=None,
        num_time_steps=200,
        return_trajectory=False,
    )

    expected = x0 + 1.0
    assert ys.shape == x0.shape
    assert jnp.allclose(ys, expected, atol=1e-3, rtol=0.0)


def test_sde_return_trajectory_shape():
    """return_trajectory=True should give [T, *y0.shape]."""
    num_time_steps = 50
    drift_vf = ConstantDriftVF(value=0.0)
    dynamics: TSDEDynamics = (drift_vf, _diff_zero_time_state)

    solver = SDESolver(dynamics=dynamics, method=dfx.Euler(), device_id="cpu")
    x0 = jnp.zeros((4,))

    traj = solver.solve(
        source=x0,
        brownian_motion=None,
        num_time_steps=num_time_steps,
        return_trajectory=True,
    )

    assert traj.shape == (num_time_steps, *x0.shape)
    assert jnp.allclose(traj, jnp.broadcast_to(x0, traj.shape), atol=1e-6, rtol=0.0)


def test_sde_with_custom_virtual_brownian_tree_runs():
    """Using an explicit VirtualBrownianTree should still solve without errors."""
    drift_vf = ConstantDriftVF(value=1.0)
    dynamics: TSDEDynamics = (drift_vf, _diff_identity_time_state)

    solver = SDESolver(dynamics=dynamics, method=dfx.Euler(), device_id="cpu")
    x0 = jnp.array([0.0, 0.0])

    key = jax.random.PRNGKey(0)
    brownian = dfx.VirtualBrownianTree(
        t0=0.0,
        t1=1.0,
        tol=1e-3,
        shape=x0.shape,
        key=key,
    )

    ys = solver.solve(
        source=x0,
        brownian_motion=brownian,
        num_time_steps=100,
        return_trajectory=False,
    )

    assert ys.shape == x0.shape
    assert jnp.all(jnp.isfinite(ys))


def test_sde_unsafe_brownian_fixed_stepsize_requires_adjoint():
    """UnsafeBrownianPath with adjoint=None should raise a clear ValueError."""
    drift_vf = ConstantDriftVF(value=1.0)
    dynamics: TSDEDynamics = (drift_vf, _diff_identity_time_state)

    solver = SDESolver(dynamics=dynamics, method=dfx.Euler(), device_id="cpu")
    x0 = jnp.array(0.0)

    brownian = dfx.UnsafeBrownianPath(
        shape=(),
        key=jax.random.PRNGKey(0),
    )

    with pytest.raises(
        ValueError,
        match="Cannot use UnsafeBrownianPath with adjoint=None",
    ):
        _ = solver.solve(
            source=x0,
            brownian_motion=brownian,
            num_time_steps=10,
            return_trajectory=False,
        )


def test_sde_unsafe_brownian_fixed_stepsize_with_forwardmode_warns_and_runs():
    """UnsafeBrownianPath + fixed-step + ForwardMode adjoint runs and emits a warning."""
    drift_vf = ConstantDriftVF(value=1.0)
    dynamics: TSDEDynamics = (drift_vf, _diff_identity_time_state)

    solver = SDESolver(dynamics=dynamics, method=dfx.Euler(), device_id="cpu")
    x0 = jnp.array(0.0)

    brownian = dfx.UnsafeBrownianPath(
        shape=(),
        key=jax.random.PRNGKey(1),
    )

    adjoint = dfx.ForwardMode()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        y_final = solver.solve(
            source=x0,
            brownian_motion=brownian,
            num_time_steps=10,
            return_trajectory=False,
            solver_kwargs={"adjoint": adjoint},
        )

    assert jnp.isfinite(y_final)
    assert any("Using UnsafeBrownianPath with fixed-step solver" in str(warn.message) for warn in w)


def test_sde_dynamics_tuple_types_and_storage():
    """Check that dynamics tuple is stored correctly in the base Solver."""
    drift_vf = ConstantDriftVF(value=1.0)
    diffusion = _diff_zero_time_state
    dynamics: TSDEDynamics = (drift_vf, diffusion)

    solver = SDESolver(dynamics=dynamics, method="euler", device_id="cpu")

    dyn = solver.dynamics
    assert isinstance(dyn, tuple)
    assert isinstance(dyn[0], BaseVelocityField)
    assert callable(dyn[1])


def test_sde_drift_and_diffusion_fn_types():
    """Check that drift_fn and diffusion_fn have the correct (t, y, args) signature."""
    drift_vf = ConstantDriftVF(value=2.0)
    dynamics: TSDEDynamics = (drift_vf, _diff_identity_time_state)

    solver = SDESolver(dynamics=dynamics, method=dfx.Euler(), device_id="cpu")

    drift_fn = solver.drift_fn
    assert callable(drift_fn)

    t = jnp.array(0.0)
    y = jnp.array([3.0])
    args = None

    drift_val = drift_fn(t, y, args)
    assert isinstance(drift_val, jnp.ndarray)
    assert drift_val.shape == y.shape
    assert jnp.allclose(drift_val, 2.0 * jnp.ones_like(y))

    diffusion_fn = solver.diffusion_fn
    assert callable(diffusion_fn)

    diff_val = diffusion_fn(t, y, args)

    assert hasattr(diff_val, "diagonal")
    diag = diff_val.diagonal
    assert isinstance(diag, jnp.ndarray)

    assert diag.shape[-1] == y.shape[-1]
    assert jnp.all(jnp.isfinite(diag))


def test_sde_time_only_diffusion_is_wrapped_to_time_state():
    """A diffusion g(t, args) should be wrapped so solver.diffusion_fn(t, y, args) still works."""
    drift_vf = ConstantDriftVF(value=0.0)
    dynamics: TSDEDynamics = (drift_vf, _diff_identity_time_only)

    solver = SDESolver(dynamics=dynamics, method=dfx.Euler(), device_id="cpu")

    t = jnp.array(0.0)
    y = jnp.array([1.0, 2.0])
    args = jnp.array([0.0, 0.0])

    diffusion_fn = solver.diffusion_fn
    diff_val = diffusion_fn(t, y, args)

    assert hasattr(diff_val, "diagonal")
    diag = diff_val.diagonal
    assert isinstance(diag, jnp.ndarray)
    assert diag.shape[-1] == y.shape[-1]
    assert jnp.all(jnp.isfinite(diag))


def test_sde_method_string_is_mapped_to_solver():
    """Passing a string method should be mapped to a concrete diffrax solver via get_sde_solver."""
    drift_vf = ConstantDriftVF(value=1.0)
    dynamics: TSDEDynamics = (drift_vf, _diff_zero_time_state)

    solver = SDESolver(dynamics=dynamics, method="euler", device_id="cpu")
    assert isinstance(solver._method, dfx.AbstractSolver)
