import json

import pytest
import torch
import torch.nn as nn

from sc_flow._registry import PortabilityError, parse
from sc_flow.flow._objective_config import (
    FlowBuildContext,
    GENOTObjectiveConfig,
    LinearDiracPathConfig,
    LinearGaussianPathConfig,
    LiveCostFn,
    OTFMObjectiveConfig,
)
from sc_flow.flow._objectives import GENOTObjective, OTFMObjective


class _StubVF(nn.Module):
    """Minimal unconditional velocity field: v(t, x) with real parameters (device/dtype come from them)."""

    is_conditional = False

    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.net = nn.Linear(dim + 1, dim)

    def velocity_from_embedding(self, t, x_t, emb):  # emb is None (unconditional)
        return self.net(torch.cat([x_t, t], dim=-1))


def test_otfm_config_roundtrips_with_nested_path():
    cfg = OTFMObjectiveConfig(
        probability_path=LinearGaussianPathConfig(sigma=0.1),
        match_method="sinkhorn",
        match_kwargs={"epsilon": 0.1, "scale_cost": "mean"},
    )
    spec = cfg.to_spec()
    assert spec["type"] == "flow.otfm"
    assert spec["config"]["probability_path"]["type"] == "flow.path.linear_gaussian"  # nested Component
    assert parse(json.loads(json.dumps(spec))) == cfg


def test_builds_the_real_runtime_objectives():
    assert isinstance(OTFMObjectiveConfig().build(FlowBuildContext(seed=0)), OTFMObjective)
    assert isinstance(GENOTObjectiveConfig().build(FlowBuildContext(seed=0)), GENOTObjective)
    # a different path is a one-field swap
    assert isinstance(
        OTFMObjectiveConfig(probability_path=LinearDiracPathConfig()).build(FlowBuildContext()), OTFMObjective
    )


def test_otfm_runs_independent_coupling():
    model = _StubVF(8)
    batch = {"source": torch.randn(24, 8), "target": torch.randn(24, 8) + 2.0}
    obj = OTFMObjectiveConfig(match_method="independent").build(FlowBuildContext(seed=0))
    loss, logs = obj.compute_loss(model, batch)
    assert torch.isfinite(loss) and "loss" in logs


def test_otfm_runs_real_jax_ott_sinkhorn():
    model = _StubVF(8)
    batch = {"source": torch.randn(24, 8), "target": torch.randn(24, 8) + 2.0}
    obj = OTFMObjectiveConfig(match_method="sinkhorn", match_kwargs={"epsilon": 0.1}).build(FlowBuildContext(seed=0))
    loss, logs = obj.compute_loss(model, batch)  # exercises the jax/ott coupling
    assert torch.isfinite(loss)


def test_live_costfn_builds_but_is_not_portable():
    cfg = OTFMObjectiveConfig(match_kwargs={"cost_fn": LiveCostFn()})
    assert isinstance(cfg.build(FlowBuildContext()), OTFMObjective)  # a live cost fn trains fine
    with pytest.raises(PortabilityError):
        cfg.to_spec()  # ...but the config that holds it cannot be serialized
