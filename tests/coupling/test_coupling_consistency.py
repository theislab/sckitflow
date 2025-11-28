import jax.numpy as jnp
import numpy as np
import torch

from sc_flow.backends.jax import coupling as coupling_jax
from sc_flow.backends.torch import coupling as coupling_torch


# ===============================================================
# Tests for ot_linear_coupling consistency
# ===============================================================
def test_linear_sinkhorn_coupling_consistency():
    np.random.seed(0)

    src = np.random.rand(1000, 10)
    tgt = np.random.rand(1000, 10)

    src_torch = torch.from_numpy(src)
    tgt_torch = torch.from_numpy(tgt)

    src_jax = jnp.asarray(src)
    tgt_jax = jnp.asarray(tgt)

    _, _, coupling_matrix_torch = coupling_torch.ot_linear_coupling(src_torch, tgt_torch, return_matrix=True)

    _, _, coupling_matrix_jax = coupling_jax.ot_linear_coupling(src_jax, tgt_jax, return_matrix=True)

    assert (np.abs(np.asarray(coupling_matrix_torch) - np.asarray(coupling_matrix_jax)).max()) < 1e-4
    assert (np.abs(np.asarray(coupling_matrix_torch) - np.asarray(coupling_matrix_jax)).mean()) < 1e-5


def test_linear_unbalanced_coupling_consistency():
    np.random.seed(0)

    src = np.random.rand(1000, 10)
    tgt = np.random.rand(1000, 10)

    src_torch = torch.from_numpy(src)
    tgt_torch = torch.from_numpy(tgt)

    src_jax = jnp.asarray(src)
    tgt_jax = jnp.asarray(tgt)

    _, _, coupling_matrix_torch = coupling_torch.ot_linear_coupling(
        src_torch, tgt_torch, method="unbalanced", return_matrix=True
    )

    _, _, coupling_matrix_jax = coupling_jax.ot_linear_coupling(
        src_jax, tgt_jax, method="unbalanced", tau_1=0.9, tau_b=0.9, return_matrix=True
    )

    assert (np.abs(np.asarray(coupling_matrix_torch) - np.asarray(coupling_matrix_jax)).max()) < 1e-4
    assert (np.abs(np.asarray(coupling_matrix_torch) - np.asarray(coupling_matrix_jax)).mean()) < 1e-5


# ===============================================================
# Tests for ot_quadratic_coupling consistency
# ===============================================================
def test_quadratic_coupling_consistency():
    np.random.seed(0)

    src_xx = np.random.rand(1000, 10)
    tgt_yy = np.random.rand(1000, 15)

    src_xx_torch = torch.from_numpy(src_xx)
    tgt_yy_torch = torch.from_numpy(tgt_yy)

    src_xx_jax = jnp.asarray(src_xx)
    tgt_yy_jax = jnp.asarray(tgt_yy)

    _, _, coupling_matrix_torch = coupling_torch.ot_quadratic_coupling(src_xx_torch, tgt_yy_torch, return_matrix=True)

    _, _, coupling_matrix_jax = coupling_jax.ot_quadratic_coupling(src_xx_jax, tgt_yy_jax, return_matrix=True)

    assert (np.abs(np.asarray(coupling_matrix_torch) - np.asarray(coupling_matrix_jax)).max()) < 1e-4
    assert (np.abs(np.asarray(coupling_matrix_torch) - np.asarray(coupling_matrix_jax)).mean()) < 1e-5


def test_quadratic_fused_coupling_consistency():
    np.random.seed(0)

    src_xx = np.random.rand(1000, 10)
    tgt_yy = np.random.rand(1000, 15)
    src_xy = np.random.rand(1000, 7)
    tgt_xy = np.random.rand(1000, 7)

    src_xx_torch = torch.from_numpy(src_xx)
    tgt_yy_torch = torch.from_numpy(tgt_yy)
    src_xy_torch = torch.from_numpy(src_xy)
    tgt_xy_torch = torch.from_numpy(tgt_xy)

    src_xx_jax = jnp.asarray(src_xx)
    tgt_yy_jax = jnp.asarray(tgt_yy)
    src_xy_jax = jnp.asarray(src_xy)
    tgt_xy_jax = jnp.asarray(tgt_xy)

    _, _, coupling_matrix_torch = coupling_torch.ot_quadratic_coupling(
        src_xx_torch,
        tgt_yy_torch,
        src_xy_torch,
        tgt_xy_torch,
        method="entropic_fused_gromov_wasserstein",
        return_matrix=True,
    )

    _, tgt_idx__jax, coupling_matrix_jax = coupling_jax.ot_quadratic_coupling(
        src_xx_jax, tgt_yy_jax, src_xy_jax, tgt_xy_jax, method="entropic_fused_gromov_wasserstein", return_matrix=True
    )

    assert (np.abs(np.asarray(coupling_matrix_torch) - np.asarray(coupling_matrix_jax)).max()) < 1e-4
    assert (np.abs(np.asarray(coupling_matrix_torch) - np.asarray(coupling_matrix_jax)).mean()) < 1e-5
