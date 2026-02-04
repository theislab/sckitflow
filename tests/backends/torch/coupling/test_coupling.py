import numpy as np
import pytest
import torch

from sc_flow.backends.torch.coupling import (
    independent_coupling,
    ot_linear_coupling,
    ot_quadratic_coupling,
)


@pytest.fixture
def small_tensors():
    source = torch.rand(5, 3)
    target = torch.rand(6, 3)
    return source, target


# ===============================================================
# Tests for independent_coupling
# ===============================================================
def test_independent_coupling_returns_indices(small_tensors):
    source, target = small_tensors
    src_idx, tgt_idx = independent_coupling(source, target)

    # Basic type and shape checks
    assert isinstance(src_idx, np.ndarray)
    assert isinstance(tgt_idx, np.ndarray)
    assert src_idx.ndim == 0 or src_idx.ndim == 1
    assert tgt_idx.ndim == 0 or tgt_idx.ndim == 1
    assert len(src_idx) == len(tgt_idx)
    assert (src_idx >= 0).all() and (src_idx < len(source)).all()
    assert (tgt_idx >= 0).all() and (tgt_idx < len(target)).all()


def test_independent_coupling_reproducibility(monkeypatch, small_tensors):
    """Ensure deterministic behavior if random seed is fixed."""
    source, target = small_tensors
    np.random.seed(42)
    idx1 = independent_coupling(source, target)
    np.random.seed(42)
    idx2 = independent_coupling(source, target)
    assert all(np.allclose(a, b) for a, b in zip(idx1, idx2, strict=False))


# ===============================================================
# Tests for ot_linear_coupling
# ===============================================================
@pytest.mark.parametrize("method", ["exact", "sinkhorn"])
def test_ot_linear_coupling_basic(method, small_tensors):
    source, target = small_tensors
    src_idx, tgt_idx = ot_linear_coupling(source, target, method=method)

    assert isinstance(src_idx, np.ndarray)
    assert isinstance(tgt_idx, np.ndarray)
    assert len(src_idx) == len(source)
    assert len(tgt_idx) == len(source)
    assert (src_idx >= 0).all() and (src_idx < len(source)).all()
    assert (tgt_idx >= 0).all() and (tgt_idx < len(target)).all()


def test_ot_linear_coupling_invalid_method_raises(small_tensors):
    source, target = small_tensors
    with pytest.raises(ValueError):
        ot_linear_coupling(source, target, method="invalid", reg=5e-1)


@pytest.mark.parametrize("scale_cost", ["mean", "max", "median", 1.0])
def test_ot_linear_coupling_scale_modes(scale_cost, small_tensors):
    source, target = small_tensors
    src_idx, tgt_idx = ot_linear_coupling(source, target, scale_cost=scale_cost, method="sinkhorn")

    dim_out = min(len(src_idx), len(tgt_idx))
    assert isinstance(src_idx, np.ndarray)
    assert len(src_idx) == dim_out
    assert isinstance(tgt_idx, np.ndarray)
    assert len(tgt_idx) == dim_out


# ===============================================================
# Tests for ot_quadratic_coupling
# ===============================================================
@pytest.fixture
def quadratic_inputs():
    torch.manual_seed(0)
    src_xx = torch.rand(4, 3)
    tgt_yy = torch.rand(5, 3)
    src_xy = torch.rand(4, 3)
    tgt_xy = torch.rand(5, 3)
    return src_xx, tgt_yy, src_xy, tgt_xy


@pytest.mark.parametrize("method", ["entropic_gromov_wasserstein", "entropic_fused_gromov_wasserstein"])
def test_ot_quadratic_coupling_basic(quadratic_inputs, method):
    src_xx, tgt_yy, src_xy, tgt_xy = quadratic_inputs
    src_idx, tgt_idx = ot_quadratic_coupling(
        source_quad=src_xx,
        target_quad=tgt_yy,
        source_lin=src_xy,
        target_lin=tgt_xy,
        method=method,
    )

    dim_out = min(len(src_xx), len(tgt_yy))
    assert isinstance(src_idx, np.ndarray)
    assert isinstance(tgt_idx, np.ndarray)
    assert isinstance(src_idx, np.ndarray)
    assert len(src_idx) == dim_out
    assert isinstance(tgt_idx, np.ndarray)
    assert len(tgt_idx) == dim_out
    assert (src_idx >= 0).all() and (tgt_idx >= 0).all()


def test_ot_quadratic_coupling_without_xy_couplings(quadratic_inputs):
    src_xx, tgt_yy, *_ = quadratic_inputs
    src_idx, tgt_idx = ot_quadratic_coupling(
        source_quad=src_xx,
        target_quad=tgt_yy,
        method="entropic_gromov_wasserstein",
    )

    dim_out = min(len(src_xx), len(tgt_yy))
    assert isinstance(src_idx, np.ndarray)
    assert len(src_idx) == dim_out
    assert isinstance(tgt_idx, np.ndarray)
    assert len(tgt_idx) == dim_out


def test_ot_quadratic_coupling_invalid_method_raises(quadratic_inputs):
    src_xx, tgt_yy, *_ = quadratic_inputs
    with pytest.raises(ValueError):
        ot_quadratic_coupling(source_quad=src_xx, target_quad=tgt_yy, method="invalid_method")
