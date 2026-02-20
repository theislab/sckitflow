import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sc_flow.backends.jax.coupling import (
    independent_coupling,
    ot_linear_coupling,
    ot_quadratic_coupling,
)

seed = 0

# -------------------------------------------------------------------------
# Tests for independent_coupling
# -------------------------------------------------------------------------


def test_independent_coupling_shapes():
    np.random.seed(0)

    src = jnp.arange(10)
    tgt = jnp.arange(20)

    src_idx, tgt_idx = independent_coupling(
        src,
        tgt,
    )

    assert len(src_idx) == len(tgt_idx)
    assert isinstance(src_idx, np.ndarray)
    assert isinstance(tgt_idx, np.ndarray)


def test_independent_coupling_values_within_bounds():
    np.random.seed(1)

    src = jnp.arange(5)
    tgt = jnp.arange(7)

    src_idx, tgt_idx = independent_coupling(src, tgt)

    assert np.all(src_idx < len(src))
    assert np.all(tgt_idx < len(tgt))


# -------------------------------------------------------------------------
# Tests for ot_linear_coupling
# -------------------------------------------------------------------------


def test_ot_linear_coupling_basic():
    np.random.seed(0)
    key = jax.random.key(0)
    key, subkey_1, subkey_2 = jax.random.split(key, num=3)

    src = jax.random.normal(shape=(5, 2), key=subkey_1)
    tgt = jax.random.normal(shape=(6, 2), key=subkey_2)

    src_idx, tgt_idx = ot_linear_coupling(src, tgt)

    # Output validity
    assert len(src_idx) == src.shape[0]
    assert len(tgt_idx) == src.shape[0]

    # Index bounds
    assert np.all(src_idx < src.shape[0])
    assert np.all(tgt_idx < tgt.shape[0])


def test_ot_linear_coupling_exact_vs_sinkhorn_same_shape():
    np.random.seed(0)
    key = jax.random.key(0)

    key, subkey_1, subkey_2 = jax.random.split(key, num=3)

    src = jax.random.normal(shape=(4, 3), key=subkey_1)
    tgt = jax.random.normal(shape=(5, 3), key=subkey_2)

    out_sinkhorn = ot_linear_coupling(src, tgt, method="sinkhorn")

    assert len(out_sinkhorn[0]) == src.shape[0]


def test_ot_linear_coupling_partial_raises():
    key = jax.random.key(0)
    key, subkey_1, subkey_2 = jax.random.split(key, num=3)

    src = jax.random.normal(shape=(4, 2), key=subkey_1)
    tgt = jax.random.normal(shape=(4, 2), key=subkey_2)

    with pytest.raises(ValueError):
        ot_linear_coupling(src, tgt, method="partial")


# -------------------------------------------------------------------------
# Tests for ot_quadratic_coupling
# -------------------------------------------------------------------------


def test_ot_quadratic_coupling_basic():
    np.random.seed(0)
    key = jax.random.key(0)
    key, subkey_1 = jax.random.split(key, num=2)

    xx = np.random.rand(5, 5)
    yy = np.random.rand(6, 6)

    src_idx, tgt_idx = ot_quadratic_coupling(
        source_quad=xx,
        target_quad=yy,
    )

    assert len(src_idx) == len(tgt_idx) == xx.shape[0]
    assert np.all(src_idx < xx.shape[0])
    assert np.all(tgt_idx < yy.shape[0])


def test_ot_quadratic_requires_fused_terms():
    key = jax.random.key(0)
    key, subkey_1 = jax.random.split(key, num=2)

    xx = np.random.rand(3, 3)
    yy = np.random.rand(4, 4)

    # Missing fused terms → must fail
    with pytest.raises(ValueError):
        ot_quadratic_coupling(
            xx,
            yy,
            method="entropic_fused_gromov_wasserstein",
        )


def test_ot_quadratic_coupling_fused_terms():
    np.random.seed(0)
    key = jax.random.key(0)

    key, subkey_1 = jax.random.split(key, num=2)

    src = jax.random.normal(shape=(4, 2), key=subkey_1)

    xx = np.random.rand(4, 4)
    yy = np.random.rand(5, 5)
    xy = np.random.rand(4, 3)
    yx = np.random.rand(5, 3)

    src_idx, tgt_idx = ot_quadratic_coupling(
        xx,
        yy,
        source_lin=xy,
        target_lin=yx,
        method="entropic_fused_gromov_wasserstein",
    )

    assert len(src_idx) == src.shape[0]
    assert np.all(src_idx < xx.shape[0])
    assert np.all(tgt_idx < yy.shape[0])
