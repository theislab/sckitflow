import warnings

import diffrax as dfx
import jax
import jax.numpy as jnp
import lineax as lx
import pytest

from sc_flow.backends.jax.solvers.sde_solver import SDESolver


def _drift_const_one(t, y, args=None):
    return jnp.ones_like(y)


def _drift_zero(t, y, args=None):
    return jnp.zeros_like(y)


def _diff_zero(t, y, args=None):
    return lx.DiagonalLinearOperator(jnp.zeros_like(y))


def _diff_identity(t, y, args=None):
    return lx.DiagonalLinearOperator(jnp.ones_like(y))


def test_sde_deterministic_zero_diffusion_matches_ode():
    """dX = 1 dt, sigma = 0  =>  X(1) = X(0) + 1."""
    solver = SDESolver(method=dfx.Euler(), num_time_steps=200, device_id="cpu")
    x0 = jnp.array([0.0, 2.0, -1.0])

    ys = solver.solve(
        source=x0,
        drift_fn=_drift_const_one,
        diffusion_fn=_diff_zero,
        brownian_motion=None,
        return_trajectory=False,
        solver_kwargs={},
    )

    expected = x0 + 1.0
    assert ys.shape == x0.shape
    assert jnp.allclose(ys, expected, atol=1e-3, rtol=0.0)


def test_sde_return_trajectory_shape():
    """return_trajectory=True should give [T, *y0.shape]."""
    num_time_steps = 50
    solver = SDESolver(method=dfx.Euler(), num_time_steps=num_time_steps, device_id="cpu")
    x0 = jnp.zeros((4,))

    traj = solver.solve(
        source=x0,
        drift_fn=_drift_zero,
        diffusion_fn=_diff_zero,
        brownian_motion=None,
        return_trajectory=True,
        solver_kwargs={},
    )

    assert traj.shape == (num_time_steps, *x0.shape)
    assert jnp.allclose(traj, jnp.broadcast_to(x0, traj.shape), atol=1e-6, rtol=0.0)


def test_sde_with_custom_virtual_brownian_tree_runs():
    """Using an explicit VirtualBrownianTree should still solve without errors."""
    solver = SDESolver(method=dfx.Euler(), num_time_steps=100, device_id="cpu")
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
        drift_fn=_drift_const_one,
        diffusion_fn=_diff_identity,
        brownian_motion=brownian,
        return_trajectory=False,
        solver_kwargs={},
    )

    assert ys.shape == x0.shape
    assert jnp.all(jnp.isfinite(ys))


def test_sde_unsafe_brownian_fixed_stepsize_without_adjoint_raises() -> None:
    """UnsafeBrownianPath + fixed-step + no adjoint must raise ValueError."""
    solver = SDESolver(method=dfx.Euler(), num_time_steps=10, device_id="cpu")
    x0 = jnp.array(0.0)

    brownian = dfx.UnsafeBrownianPath(
        shape=(),
        key=jax.random.PRNGKey(0),
    )

    def drift(t, y, args):
        return jnp.ones_like(y)

    def diffusion(t, y, args):
        return jnp.ones_like(y)

    with pytest.raises(
        ValueError,
    ):
        _ = solver.solve(
            source=x0,
            drift_fn=drift,
            diffusion_fn=diffusion,
            brownian_motion=brownian,
            return_trajectory=False,
            solver_kwargs={},
        )


def test_sde_unsafe_brownian_fixed_stepsize_with_forwardmode_adjoint_warns_and_runs() -> None:
    """UnsafeBrownianPath + fixed-step + ForwardModeAdjoint runs and emits a warning."""
    solver = SDESolver(method=dfx.Euler(), num_time_steps=10, device_id="cpu")
    x0 = jnp.array(0.0)

    brownian = dfx.UnsafeBrownianPath(
        shape=(),
        key=jax.random.PRNGKey(1),
    )

    def drift(t, y, args):
        return 1.0

    def diffusion(t, y, args):
        return 0.0

    adjoint = dfx.ForwardMode()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        y_final = solver.solve(
            source=x0,
            drift_fn=drift,
            diffusion_fn=diffusion,
            brownian_motion=brownian,
            return_trajectory=False,
            solver_kwargs={"adjoint": adjoint},
        )

    assert jnp.isfinite(y_final)
    assert float(y_final) > 0.0

    assert any("Using UnsafeBrownianPath with fixed-step solver" in str(warn.message) for warn in w)
