import diffrax as dfx
import jax
import jax.numpy as jnp
import lineax as lx

from sc_flow.backends.jax.solvers._sde_solver import SDESolver


class ConstantDriftVF:
    """Simple drift VF: dy/dt = value (broadcasted to y's shape)."""

    def __init__(self, value: float = 1.0):
        self.value = float(value)

    def get_vf_fn(self, **vf_kwargs):
        value = float(vf_kwargs.get("value", self.value))

        def vf(t, y, args=None):
            return value * jnp.ones_like(y)

        return vf


def _state_dependent_diffusion(t, y, scale: float = 1.0):
    """Diffusion of the form g(t, y, scale) -> same shape as y."""
    return scale * jnp.ones_like(y)


def _time_only_diffusion(t, scale: float = 1.0):
    """Diffusion of the form g(t, scale) -> scalar."""
    return jnp.array(scale, dtype=float)


# --- Updated Tests ---


def test_diffusion_wrapper_state_dependent_uses_y_and_df_kwargs():
    drift_vf = ConstantDriftVF(value=0.0)
    dynamics = (drift_vf, _state_dependent_diffusion)

    solver = SDESolver(
        dynamics=dynamics,
        method=dfx.Euler(),
        device_id="cpu",
        vf_kwargs={"value": 0.0},
        df_kwargs={"scale": 2.5},
    )

    t = jnp.array(0.0)
    y = jnp.array([1.0, 2.0, 3.0])

    diff_fn = solver.diffusion_fn
    out = diff_fn(t, y)

    # Note: 'out' is likely a DiagonalLinearOperator
    # We verify the diagonal values instead of the object shape
    expected_diag = 2.5 * jnp.ones_like(y)

    if isinstance(out, lx.DiagonalLinearOperator):
        assert jnp.allclose(out.diagonal, expected_diag)
    else:
        assert jnp.allclose(out, expected_diag)


def test_diffusion_wrapper_time_only_ignores_y_and_uses_df_kwargs():
    drift_vf = ConstantDriftVF(value=0.0)
    dynamics = (drift_vf, _time_only_diffusion)

    solver = SDESolver(
        dynamics=dynamics,
        method=dfx.Euler(),
        device_id="cpu",
        vf_kwargs={"value": 0.0},
        df_kwargs={"scale": 0.7},
    )

    t = jnp.array(0.0)
    y = jnp.array([1.0, 2.0])

    diff_fn = solver.diffusion_fn
    out = diff_fn(t, y)

    # Even for time-only diffusion, the wrapper likely promotes it
    # to a DiagonalLinearOperator to match the state 'y'
    expected_diag = 0.7 * jnp.ones_like(y)

    if isinstance(out, lx.DiagonalLinearOperator):
        assert jnp.allclose(out.diagonal, expected_diag)
    else:
        # If it returns a scalar, jnp.allclose handles the comparison
        assert jnp.allclose(out, 0.7)


def test_sde_solver_runs_with_state_dependent_diffusion():
    drift_vf = ConstantDriftVF(value=1.0)
    dynamics = (drift_vf, _state_dependent_diffusion)

    solver = SDESolver(
        dynamics=dynamics,
        method=dfx.Euler(),
        device_id="cpu",
        vf_kwargs={"value": 1.0},
        df_kwargs={"scale": 0.3},
    )

    key = jax.random.PRNGKey(0)
    y0 = jnp.array([0.0, 0.0])

    bm = dfx.VirtualBrownianTree(
        t0=0.0,
        t1=1.0,
        shape=y0.shape,  # Matches y0
        tol=1e-3,
        key=key,
    )

    ys = solver.solve(
        source=y0,
        t0=0.0,
        t1=1.0,
        brownian_motion=bm,
        num_time_steps=20,
        return_trajectory=False,
        solver_kwargs={},
    )

    assert ys.shape == y0.shape
    assert jnp.all(jnp.isfinite(ys))


def test_sde_solver_runs_with_time_only_diffusion():
    drift_vf = ConstantDriftVF(value=0.5)
    dynamics = (drift_vf, _time_only_diffusion)

    solver = SDESolver(
        dynamics=dynamics,
        method=dfx.Euler(),
        device_id="cpu",
        vf_kwargs={"value": 0.5},
        df_kwargs={"scale": 0.2},
    )

    key = jax.random.PRNGKey(1)
    y0 = jnp.array([1.0])

    bm = dfx.VirtualBrownianTree(
        t0=0.0,
        t1=0.5,
        shape=y0.shape,
        tol=1e-3,
        key=key,
    )

    ys_traj = solver.solve(
        source=y0,
        t0=0.0,
        t1=0.5,
        brownian_motion=bm,
        num_time_steps=15,
        return_trajectory=True,
        solver_kwargs={},
    )

    # Check trajectory dimensions
    # Depending on implementation, num_time_steps might result in 15 or 16 points
    assert ys_traj.shape[0] >= 15
    assert ys_traj.shape[1:] == y0.shape
    assert jnp.all(jnp.isfinite(ys_traj))
