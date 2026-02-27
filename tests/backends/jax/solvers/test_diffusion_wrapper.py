import diffrax as dfx
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

from sc_flow.backends.jax.solvers.sde_solver import SDESolver


def _wrap(diffusion_fn, df_kwargs=None):
    """Call SDESolver._get_diffusion_fn_wrapper without constructing a full solver."""
    s = SDESolver.__new__(SDESolver)  # bypass __init__
    return s._get_diffusion_fn_wrapper(diffusion_fn, df_kwargs)


def _solve_once(wrapped_diffusion, y0, *, bm_shape, t0=0.0, t1=0.2, dt0=0.02):
    """Run a minimal SDE solve in Diffrax to ensure the wrapped diffusion is accepted."""

    def drift(t, y, args):
        return jnp.zeros_like(y)

    bm = dfx.VirtualBrownianTree(
        t0=t0,
        t1=t1,
        tol=1e-3,
        shape=bm_shape,
        key=jr.PRNGKey(0),
    )

    terms = dfx.MultiTerm(
        dfx.ODETerm(drift),
        dfx.ControlTerm(wrapped_diffusion, bm),
    )

    sol = dfx.diffeqsolve(
        terms=terms,
        solver=dfx.Euler(),
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=y0,
        saveat=dfx.SaveAt(t1=True),
        stepsize_controller=dfx.ConstantStepSize(),
        max_steps=10_000,
    )

    yT = sol.ys[0]
    assert yT.shape == y0.shape
    assert jnp.all(jnp.isfinite(yT))
    return yT


@pytest.mark.parametrize("d", [3, 7])
def test_wrapped_diffusion_returns_operator_and_diffrax_accepts_scalar(d):
    y0 = jnp.linspace(0.1, 1.0, d)

    def diffusion_scalar(t):
        return jnp.array(0.1)

    wrapped = _wrap(diffusion_scalar)
    op = wrapped(0.0, y0, None)

    assert isinstance(op, lx.AbstractLinearOperator)
    assert isinstance(op, lx.DiagonalLinearOperator)

    _solve_once(wrapped, y0, bm_shape=y0.shape)


@pytest.mark.parametrize("d", [3, 7])
def test_wrapped_diffusion_vector_same_shape_becomes_diagonal_operator(d):
    y0 = jnp.linspace(0.1, 1.0, d)

    def diffusion_vector(t, y):
        return 0.05 + 0.01 * jnp.arange(y.shape[0], dtype=y.dtype)

    wrapped = _wrap(diffusion_vector)
    op = wrapped(0.0, y0, None)

    assert isinstance(op, lx.AbstractLinearOperator)
    assert isinstance(op, lx.DiagonalLinearOperator)

    _solve_once(wrapped, y0, bm_shape=y0.shape)


@pytest.mark.parametrize("d", [3, 7])
def test_wrapped_diffusion_matrix_becomes_matrix_operator_square(d):
    y0 = jnp.linspace(0.1, 1.0, d)

    def diffusion_matrix(t, y):
        return 0.1 * jnp.eye(y.shape[0], dtype=y.dtype)

    wrapped = _wrap(diffusion_matrix)
    op = wrapped(0.0, y0, None)

    assert isinstance(op, lx.AbstractLinearOperator)
    assert isinstance(op, lx.MatrixLinearOperator)

    _solve_once(wrapped, y0, bm_shape=(d,))


@pytest.mark.parametrize("d", [3, 7])
def test_wrapped_diffusion_matrix_becomes_matrix_operator_rectangular(d):
    y0 = jnp.linspace(0.1, 1.0, d)

    def diffusion_matrix_rect(t, y):
        return 0.2 * jnp.ones((y.shape[0], 1), dtype=y.dtype)

    wrapped = _wrap(diffusion_matrix_rect)
    op = wrapped(0.0, y0, None)

    assert isinstance(op, lx.AbstractLinearOperator)
    assert isinstance(op, lx.MatrixLinearOperator)

    _solve_once(wrapped, y0, bm_shape=(1,))


@pytest.mark.parametrize("d", [3, 7])
def test_wrapped_diffusion_passes_through_existing_linear_operator(d):
    y0 = jnp.linspace(0.1, 1.0, d)

    def diffusion_already_operator(t, y):
        diag = 0.3 * jnp.ones_like(y)
        return lx.DiagonalLinearOperator(diag)

    wrapped = _wrap(diffusion_already_operator)
    op = wrapped(0.0, y0, None)

    assert isinstance(op, lx.AbstractLinearOperator)
    assert isinstance(op, lx.DiagonalLinearOperator)

    _solve_once(wrapped, y0, bm_shape=y0.shape)


@pytest.mark.parametrize("d", [3, 7])
def test_wrapped_diffusion_fallback_branch_pytreelinearoperator_rejected_by_diffrax_for_bad_shape(d):
    y0 = jnp.linspace(0.1, 1.0, d)

    def diffusion_bad_shape(t, y):
        return jnp.ones((d, d, d), dtype=y.dtype)

    wrapped = _wrap(diffusion_bad_shape)
    op = wrapped(0.0, y0, None)

    assert isinstance(op, lx.AbstractLinearOperator)
    assert isinstance(op, lx.PyTreeLinearOperator)

    with pytest.raises((ValueError, TypeError)):
        _solve_once(wrapped, y0, bm_shape=(d,))
