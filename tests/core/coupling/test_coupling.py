import warnings

import numpy as np
import pytest
import torch
from pydantic import ValidationError
from scfit.registry import parse

from sckitflow.core.coupling import GromovWassersteinCouplingConfig

pytest.importorskip("ot")

from sckitflow.core.coupling import (
    independent_coupling,
    ot_linear_coupling,
    ot_quadratic_coupling,
)

# The coupling functions take their settings without defaults; the configs hold those.
LIN = {"num_iter_max": None, "reg": 0.5, "reg_m": 1.0}
QUAD = {"scale_cost": 1.0, "epsilon": 1.0, "max_iter": 1000, "tol": 1e-9, "alpha": 0.5}


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
    src_idx, tgt_idx = independent_coupling(source_lin=source, target_lin=target, rng=np.random.default_rng(0))

    # Basic type and shape checks
    assert isinstance(src_idx, np.ndarray)
    assert isinstance(tgt_idx, np.ndarray)
    assert src_idx.ndim == 0 or src_idx.ndim == 1
    assert tgt_idx.ndim == 0 or tgt_idx.ndim == 1
    assert len(src_idx) == len(tgt_idx)
    assert (src_idx >= 0).all() and (src_idx < len(source)).all()
    assert (tgt_idx >= 0).all() and (tgt_idx < len(target)).all()


def test_independent_coupling_reproducibility(small_tensors):
    """Ensure deterministic behavior if the generator is seeded."""
    source, target = small_tensors
    idx1 = independent_coupling(source_lin=source, target_lin=target, rng=np.random.default_rng(42))
    idx2 = independent_coupling(source_lin=source, target_lin=target, rng=np.random.default_rng(42))
    assert all(np.allclose(a, b) for a, b in zip(idx1, idx2, strict=False))


# ===============================================================
# Tests for ot_linear_coupling
# ===============================================================
@pytest.mark.parametrize("method", ["exact", "sinkhorn"])
def test_ot_linear_coupling_basic(method, small_tensors):
    source, target = small_tensors
    src_idx, tgt_idx = ot_linear_coupling(
        source_lin=source, target_lin=target, method=method, scale_cost="mean", rng=np.random.default_rng(0), **LIN
    )

    assert isinstance(src_idx, np.ndarray)
    assert isinstance(tgt_idx, np.ndarray)
    assert len(src_idx) == len(source)
    assert len(tgt_idx) == len(source)
    assert (src_idx >= 0).all() and (src_idx < len(source)).all()
    assert (tgt_idx >= 0).all() and (tgt_idx < len(target)).all()


def test_ot_linear_coupling_invalid_method_raises(small_tensors):
    source, target = small_tensors
    with pytest.raises(ValueError):
        ot_linear_coupling(
            source_lin=source,
            target_lin=target,
            method="invalid",
            scale_cost="mean",
            rng=np.random.default_rng(0),
            **LIN,
        )


@pytest.mark.parametrize("scale_cost", ["mean", "max", "median", 1.0])
def test_ot_linear_coupling_scale_modes(scale_cost, small_tensors):
    source, target = small_tensors
    src_idx, tgt_idx = ot_linear_coupling(
        source_lin=source,
        target_lin=target,
        scale_cost=scale_cost,
        method="sinkhorn",
        rng=np.random.default_rng(0),
        **LIN,
    )

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
        rng=np.random.default_rng(0),
        **QUAD,
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
        rng=np.random.default_rng(0),
        **QUAD,
    )

    dim_out = min(len(src_xx), len(tgt_yy))
    assert isinstance(src_idx, np.ndarray)
    assert len(src_idx) == dim_out
    assert isinstance(tgt_idx, np.ndarray)
    assert len(tgt_idx) == dim_out


def test_ot_quadratic_coupling_invalid_method_raises(quadratic_inputs):
    src_xx, tgt_yy, *_ = quadratic_inputs
    with pytest.raises(ValueError):
        ot_quadratic_coupling(
            source_quad=src_xx, target_quad=tgt_yy, method="invalid_method", rng=np.random.default_rng(0), **QUAD
        )


# ===============================================================
# Tests for the coupling configs
# ===============================================================
@pytest.mark.parametrize(
    "spec",
    [
        {"type": "coupling.independent"},
        {"type": "coupling.pot.exact", "num_iter_max": 50},
        {"type": "coupling.pot.sinkhorn", "reg": 0.1},
        {"type": "coupling.pot.partial", "m": 0.5, "num_iter_max": 50},
        {"type": "coupling.pot.unbalanced", "reg_m": 2.0},
        {"type": "coupling.pot.gromov_wasserstein", "max_iter": 5},
        {"type": "coupling.pot.fused_gromov_wasserstein", "alpha": 0.3, "max_iter": 5},
    ],
)
def test_coupling_config_builds_a_match_fn(spec, small_tensors):
    source, target = small_tensors
    match_fn = parse(spec).build()
    src_idx, tgt_idx = match_fn(
        source_lin=source, target_lin=target, source_quad=source, target_quad=target, rng=np.random.default_rng(0)
    )
    assert len(src_idx) == len(tgt_idx) == len(source)


@pytest.mark.parametrize(
    "spec",
    [
        {"type": "coupling.pot.sinkhorn", "regg": 0.1},
        {"type": "coupling.pot.exact", "reg": 0.1},
        {"type": "coupling.pot.sinkhorn", "m": 0.5},
        {"type": "coupling.pot.gromov_wasserstein", "alpha": 0.3},
    ],
)
def test_coupling_config_rejects_settings_its_solver_ignores(spec):
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        parse(spec)


def test_independent_coupling_uses_quad_reps_and_the_smaller_side():
    src_idx, tgt_idx = independent_coupling(
        source_lin=None,
        target_lin=None,
        source_quad=torch.rand(6, 2),
        target_quad=torch.rand(4, 2),
        rng=np.random.default_rng(0),
    )
    assert len(src_idx) == len(tgt_idx) == 4
    assert len(set(src_idx)) == 4 and src_idx.max() < 6 and tgt_idx.max() < 4


def test_ot_linear_coupling_requires_linear_reps():
    with pytest.raises(ValueError, match="requires linear source and target"):
        ot_linear_coupling(
            source_lin=None,
            target_lin=None,
            source_quad=torch.rand(4, 2),
            target_quad=torch.rand(4, 2),
            method="sinkhorn",
            scale_cost="mean",
            rng=np.random.default_rng(0),
            **LIN,
        )


def _float32_reps():
    rng = np.random.default_rng(0)
    return rng.normal(size=(40, 4)).astype("float32"), rng.normal(size=(40, 6)).astype("float32")


def test_gromov_wasserstein_defaults_find_a_plan_on_float32():
    source, target = _float32_reps()
    match = GromovWassersteinCouplingConfig().build()
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always", RuntimeWarning)
        match(source_quad=source, target_quad=target, rng=np.random.default_rng(0))
    assert not [w for w in record if issubclass(w.category, RuntimeWarning)]


def test_an_all_zero_plan_warns_that_the_batch_is_random():
    source, target = _float32_reps()  # unscaled float32 costs make entropic GW underflow
    match = GromovWassersteinCouplingConfig(scale_cost=1.0, epsilon=1.0).build()
    with pytest.warns(RuntimeWarning, match="matched uniformly at random"):
        match(source_quad=source, target_quad=target, rng=np.random.default_rng(0))
