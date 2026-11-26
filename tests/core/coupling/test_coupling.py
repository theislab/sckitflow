import numpy as np
import ot as pot
import pytest
import torch
from modelbinder.registry import parse
from pydantic import ValidationError

from sckitflow.core.coupling import FusedGromovWassersteinCouplingConfig, GromovWassersteinCouplingConfig

pytest.importorskip("ot")

from sckitflow.core.coupling import (
    independent_coupling,
    ot_linear_coupling,
    ot_quadratic_coupling,
)
from sckitflow.core.coupling._coupling import _sanitize_coupling_matrix, _scale_distance_matrix, _select_indices

# The coupling functions take their settings without defaults; the configs hold those.
LIN = {"num_iter_max": None, "reg": 0.5, "reg_m": 1.0}
# loose tolerances: at POT's defaults (inner stopThr=1e-9) float32 never converges and every solve hits the caps
QUAD = {"scale_cost": 1.0, "epsilon": 1.0, "max_iter": 100, "tol": 1e-5, "alpha": 0.5, "inner_stop_thr": 1e-5}


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
    src_idx, tgt_idx = independent_coupling(
        source_lin=source, target_lin=target, generator=torch.Generator().manual_seed(0)
    )

    # Basic type and shape checks
    assert isinstance(src_idx, torch.Tensor)
    assert isinstance(tgt_idx, torch.Tensor)
    assert src_idx.ndim == 0 or src_idx.ndim == 1
    assert tgt_idx.ndim == 0 or tgt_idx.ndim == 1
    assert len(src_idx) == len(tgt_idx)
    assert (src_idx >= 0).all() and (src_idx < len(source)).all()
    assert (tgt_idx >= 0).all() and (tgt_idx < len(target)).all()


def test_independent_coupling_reproducibility(small_tensors):
    """Ensure deterministic behavior if the generator is seeded."""
    source, target = small_tensors
    idx1 = independent_coupling(source_lin=source, target_lin=target, generator=torch.Generator().manual_seed(42))
    idx2 = independent_coupling(source_lin=source, target_lin=target, generator=torch.Generator().manual_seed(42))
    assert all(torch.equal(a, b) for a, b in zip(idx1, idx2, strict=True))


# ===============================================================
# Tests for ot_linear_coupling
# ===============================================================
@pytest.mark.parametrize("method", ["exact", "sinkhorn"])
def test_ot_linear_coupling_basic(method, small_tensors):
    source, target = small_tensors
    src_idx, tgt_idx = ot_linear_coupling(
        source_lin=source,
        target_lin=target,
        method=method,
        scale_cost="mean",
        generator=torch.Generator().manual_seed(0),
        **LIN,
    )

    assert isinstance(src_idx, torch.Tensor)
    assert isinstance(tgt_idx, torch.Tensor)
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
            generator=torch.Generator().manual_seed(0),
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
        generator=torch.Generator().manual_seed(0),
        **LIN,
    )

    dim_out = min(len(src_idx), len(tgt_idx))
    assert isinstance(src_idx, torch.Tensor)
    assert len(src_idx) == dim_out
    assert isinstance(tgt_idx, torch.Tensor)
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
        generator=torch.Generator().manual_seed(0),
        **QUAD,
    )

    dim_out = min(len(src_xx), len(tgt_yy))
    assert isinstance(src_idx, torch.Tensor)
    assert isinstance(tgt_idx, torch.Tensor)
    assert isinstance(src_idx, torch.Tensor)
    assert len(src_idx) == dim_out
    assert isinstance(tgt_idx, torch.Tensor)
    assert len(tgt_idx) == dim_out
    assert (src_idx >= 0).all() and (tgt_idx >= 0).all()


def test_ot_quadratic_coupling_without_xy_couplings(quadratic_inputs):
    src_xx, tgt_yy, *_ = quadratic_inputs
    src_idx, tgt_idx = ot_quadratic_coupling(
        source_quad=src_xx,
        target_quad=tgt_yy,
        method="entropic_gromov_wasserstein",
        generator=torch.Generator().manual_seed(0),
        **QUAD,
    )

    dim_out = min(len(src_xx), len(tgt_yy))
    assert isinstance(src_idx, torch.Tensor)
    assert len(src_idx) == dim_out
    assert isinstance(tgt_idx, torch.Tensor)
    assert len(tgt_idx) == dim_out


def test_ot_quadratic_coupling_invalid_method_raises(quadratic_inputs):
    src_xx, tgt_yy, *_ = quadratic_inputs
    with pytest.raises(ValueError):
        ot_quadratic_coupling(
            source_quad=src_xx,
            target_quad=tgt_yy,
            method="invalid_method",
            generator=torch.Generator().manual_seed(0),
            **QUAD,
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
        source_lin=source,
        target_lin=target,
        source_quad=source,
        target_quad=target,
        generator=torch.Generator().manual_seed(0),
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
        generator=torch.Generator().manual_seed(0),
    )
    assert len(src_idx) == len(tgt_idx) == 4
    assert len(src_idx.unique()) == 4 and src_idx.max() < 6 and tgt_idx.max() < 4


def test_ot_linear_coupling_requires_linear_reps():
    with pytest.raises(ValueError, match="requires linear source and target"):
        ot_linear_coupling(
            source_lin=None,
            target_lin=None,
            source_quad=torch.rand(4, 2),
            target_quad=torch.rand(4, 2),
            method="sinkhorn",
            scale_cost="mean",
            generator=torch.Generator().manual_seed(0),
            **LIN,
        )


def _float32_reps():
    rng = np.random.default_rng(0)
    return rng.normal(size=(40, 4)).astype("float32"), rng.normal(size=(40, 6)).astype("float32")


def test_gromov_wasserstein_defaults_find_a_plan_on_float32():
    source, target = _float32_reps()
    match = GromovWassersteinCouplingConfig().build()
    *_, plan = match(
        source_quad=source, target_quad=target, generator=torch.Generator().manual_seed(0), return_matrix=True
    )
    assert torch.isfinite(plan).all() and abs(plan.sum().item() - 1) < 1e-3


@pytest.mark.parametrize(
    "plan", [torch.zeros(2, 3), torch.full((2, 3), float("nan")), torch.tensor([[1.0, 0, 0], [0, float("inf"), 0]])]
)
def test_a_degenerate_plan_falls_back_to_uniform(plan):
    torch.testing.assert_close(_sanitize_coupling_matrix(plan), torch.full((2, 3), 1 / 6))


@pytest.mark.parametrize(
    "plan",
    [
        [[0.45, 0.05], [0.05, 0.45]],
        [[0.20, 0.10, 0.03], [0.02, 0.25, 0.05], [0.10, 0.05, 0.20]],
        [[0.30, 0.03], [0.05, 0.22], [0.15, 0.25]],
    ],
)
def test_sampled_pairs_follow_the_plan(plan):
    plan = torch.tensor(plan, dtype=torch.float64)
    src, tgt = _select_indices(plan, 40000, generator=torch.Generator().manual_seed(0))
    counts = torch.zeros_like(plan).index_put_((src, tgt), torch.ones_like(src, dtype=plan.dtype), accumulate=True)
    torch.testing.assert_close(counts / counts.sum(), plan, atol=0.008, rtol=0)


def test_sampling_takes_plans_past_multinomials_limit():
    plan = torch.zeros(4097, 4097)  # 2**24 + 8193 entries
    plan[4096, 4095] = 1.0
    src, tgt = _select_indices(plan, 5, generator=torch.Generator().manual_seed(0))
    assert src.tolist() == [4096] * 5 and tgt.tolist() == [4095] * 5


def _match_on(device):
    gen = torch.Generator(device=device).manual_seed(0)
    source, target = torch.rand(8, 3, generator=gen, device=device), torch.rand(9, 3, generator=gen, device=device)
    lin = ot_linear_coupling(
        source_lin=source,
        target_lin=target,
        method="sinkhorn",
        scale_cost="mean",
        return_matrix=True,
        generator=torch.Generator(device=device).manual_seed(1),
        **LIN,
    )
    quad = ot_quadratic_coupling(
        source_quad=source,
        target_quad=target,
        method="entropic_gromov_wasserstein",
        return_matrix=True,
        generator=torch.Generator(device=device).manual_seed(1),
        **{**QUAD, "scale_cost": "mean"},
    )
    return lin, quad


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")),
    ],
)
def test_ot_couplings_stay_in_torch_on_the_inputs_device(device):
    first, second = _match_on(device), _match_on(device)
    for out in first:
        assert all(isinstance(t, torch.Tensor) and t.device.type == device for t in out)
        assert not out[0].is_floating_point() and not out[1].is_floating_point()
    # a seeded generator gives the same pairs
    for a, b in zip(first, second, strict=True):
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_plan_sampling_does_not_sync_with_the_host():
    cost = torch.rand(64, 48, device="cuda")
    generator = torch.Generator(device="cuda").manual_seed(0)
    torch.cuda.set_sync_debug_mode("error")
    try:
        for scale in ("mean", "max", "median", 2.0):
            _scale_distance_matrix(cost, scale_method=scale)
        plan = _sanitize_coupling_matrix(torch.softmax(-cost.flatten(), 0).view_as(cost))
        src, tgt = _select_indices(plan, 64, generator=generator)
        cost[src, tgt]  # the indices index the batch on device
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert src.is_cuda and tgt.is_cuda


@pytest.mark.parametrize("config", [GromovWassersteinCouplingConfig, FusedGromovWassersteinCouplingConfig])
def test_quadratic_inner_sinkhorn_settings_reach_pot(config, monkeypatch, small_tensors):
    seen = {}

    def solver(*, C1, C2, M=None, **kwargs):
        seen.update(kwargs)
        return torch.full((C1.shape[0], C2.shape[0]), 1.0)

    monkeypatch.setattr(pot.gromov, config.method, solver)
    source, target = small_tensors
    match = config(inner_num_iter_max=7, inner_stop_thr=1e-4).build()
    match(source_lin=source, target_lin=target, source_quad=source, target_quad=target, generator=torch.Generator())
    assert seen["numItermax"] == 7 and seen["stopThr"] == 1e-4
    seen.clear()
    config().build()(
        source_lin=source, target_lin=target, source_quad=source, target_quad=target, generator=torch.Generator()
    )
    assert "numItermax" not in seen and "stopThr" not in seen
