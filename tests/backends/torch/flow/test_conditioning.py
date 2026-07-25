"""CFG (classifier-free guidance) + adaLN-zero conditioning — the sc-flow-tools side of the feature.

Ported from CellFlow2 (theislab/CellFlow2 @ dedup/reexport-cellflow): the zero-initialized adaLN-zero
block, the conditioning-mode selector, the learned/zero null condition + train-time dropout, and the
inference guidance blend v = v_uncond + w*(v_cond - v_uncond) with a validation scale sweep.
"""
from __future__ import annotations

import torch

from sc_flow._model import FlowMatchingConfig
from sc_flow.flow._combiner import build_combiner, validate_combiner_spec
from sc_flow.flow._config import (
    MLPEmbedderConfig,
    MLPVelocityConfig,
    SetEncoderConfig,
    VelocityFieldContext,
)
from sc_flow.flow._predict import integrate_translation
from sc_flow.flow._validation import PerturbationValidationCallback
from sc_flow.nn import AdaLNZero1d

_MEAN_POOL = {"type": "sc_flow.mean", "version": 1, "config": {}}


def _concat(latent_condition_dim=5, **cfg):
    spec = {"type": "sc_flow.concat", "version": 1, "config": cfg}
    return build_combiner(
        validate_combiner_spec(spec), latent_state_dim=6, latent_time_dim=4,
        latent_condition_dim=latent_condition_dim,
    )


def _adaln(latent_condition_dim=5, **cfg):
    spec = {"type": "sc_flow.adaln", "version": 1, "config": {"num_blocks": 2, **cfg}}
    return build_combiner(
        validate_combiner_spec(spec), latent_state_dim=6, latent_time_dim=4,
        latent_condition_dim=latent_condition_dim,
    )


def _conditional_vf(conditioning, *, condition_dropout_prob=0.0, condition_null="zero_embedding", state_dim=10):
    realm = {"type": "sc_flow.feature_mlp", "version": 1, "config": {"input_dim": 3, "output_dim": 8, "mlp_kwargs": {}}}
    ce = SetEncoderConfig(realms={"drug": realm}, output_dim=8, pooling=_MEAN_POOL)
    combiner = (
        {"type": "sc_flow.adaln", "version": 1, "config": {"num_blocks": 2}}
        if conditioning == "adaln_zero"
        else {"type": "sc_flow.concat", "version": 1, "config": {}}
    )
    cfg = MLPVelocityConfig(
        state_dim=state_dim, combiner=combiner,
        state_embedder=MLPEmbedderConfig(output_dim=6, mlp_kwargs={"hidden_dims": [8]}),
        time_embedder=MLPEmbedderConfig(output_dim=4, mlp_kwargs={}), num_time_features=4,
        condition_encoder=ce, condition_dropout_prob=condition_dropout_prob, condition_null=condition_null,
    )
    return cfg.build(VelocityFieldContext())


# --- adaLN-zero block -----------------------------------------------------------------------------


def test_adaln_zero_is_identity_at_init():
    torch.manual_seed(0)
    net = AdaLNZero1d(dim=16, cond_dim=8, num_blocks=4)
    x, cond = torch.randn(5, 16), torch.randn(5, 8)
    # zero-init modulation => scale=shift=gate=0 => every block passes x straight through.
    assert torch.allclose(net(x, cond), x, atol=1e-6)


def test_adaln_zero_activates_after_a_step():
    torch.manual_seed(0)
    net = AdaLNZero1d(dim=16, cond_dim=8, num_blocks=2)
    x, cond = torch.randn(5, 16), torch.randn(5, 8)
    opt = torch.optim.SGD(net.parameters(), lr=1.0)
    opt.zero_grad()
    ((net(x, cond) - torch.randn(5, 16)) ** 2).mean().backward()
    opt.step()
    assert not torch.allclose(net(x, cond), x, atol=1e-6)


def test_adaln_zero_preserves_shape_over_leading_dims():
    net = AdaLNZero1d(dim=7, cond_dim=3, num_blocks=2)
    x, cond = torch.randn(2, 5, 7), torch.randn(2, 5, 3)
    assert net(x, cond).shape == (2, 5, 7)


# --- combiners ------------------------------------------------------------------------------------


def test_concat_layer_norm_off_is_param_free():
    off = _concat(layer_norm=False)
    assert sum(p.numel() for p in off.parameters()) == 0  # legacy concat path stays weightless


def test_concat_layer_norm_on_adds_params_same_shape():
    et, es, ec = torch.randn(3, 4), torch.randn(3, 6), torch.randn(3, 5)
    on = _concat(layer_norm=True)
    assert sum(p.numel() for p in on.parameters()) > 0
    assert on(et, es, ec).shape == (3, 6 + 4 + 5)


def test_adaln_combiner_outputs_state_width_and_is_identity_at_init():
    et, es, ec = torch.randn(3, 4), torch.randn(3, 6), torch.randn(3, 5)
    ca = _adaln()
    out = ca(et, es, ec)
    assert out.shape == (3, 6)  # width == latent_state_dim
    assert torch.allclose(out, es, atol=1e-6)  # identity at init


def test_adaln_combiner_unconditional_uses_time_only_modulation():
    # No condition realm => modulation is time-only; still builds and preserves width.
    et, es = torch.randn(3, 4), torch.randn(3, 6)
    ca = _adaln(latent_condition_dim=None)
    assert ca(et, es, None).shape == (3, 6)


# --- config round-trips ---------------------------------------------------------------------------


def test_mlp_velocity_config_roundtrips_cfg_fields():
    vf = _conditional_vf("adaln_zero", condition_dropout_prob=0.15, condition_null="mask_value")
    d = vf.to_config().to_dict()
    assert d["condition_dropout_prob"] == 0.15 and d["condition_null"] == "mask_value"
    rebuilt = MLPVelocityConfig.from_dict(d)  # round-trips
    assert rebuilt.condition_dropout_prob == 0.15
    MLPVelocityConfig.from_spec(vf.to_config().to_spec())  # spec envelope parses back


def test_flow_matching_config_accepts_and_roundtrips_new_fields():
    fc = FlowMatchingConfig.from_dict({
        "pooling": _MEAN_POOL, "conditioning": "adaln_zero", "conditioning_kwargs": {"mlp_ratio": 4},
        "layer_norm_before_concatenation": True, "condition_dropout_prob": 0.1,
        "condition_null": "mask_value", "guidance_scale": 1.5, "guidance_scales": [1.0, 1.5, 2.0],
    })
    d = fc.to_dict()
    assert d["conditioning"] == "adaln_zero" and d["guidance_scales"] == [1.0, 1.5, 2.0]
    FlowMatchingConfig.from_dict(d)


def test_flow_matching_config_rejects_unknown_field():
    import pytest

    with pytest.raises(ValueError, match="Unknown FlowMatchingConfig field"):
        FlowMatchingConfig.from_dict({"pooling": _MEAN_POOL, "bogus": 1})


# --- CFG null path (force_uncond) -----------------------------------------------------------------


def test_force_uncond_zero_embedding_equals_zero_condition():
    torch.manual_seed(0)
    vf = _conditional_vf("concatenation", condition_dropout_prob=0.1, condition_null="zero_embedding")
    t, x, cond = torch.rand(4, 1), torch.randn(4, 10), {"drug": torch.randn(4, 1, 3)}
    v_uncond = vf(t, x, cond, force_uncond=True)
    # zero_embedding null == running the field with a zeroed pooled embedding.
    embed_dim = vf.condition_stats(cond)[0].shape[-1]
    v_zero = vf.velocity_from_embedding(t, x, torch.zeros(4, embed_dim))
    assert torch.allclose(v_uncond, v_zero, atol=1e-6)
    assert not torch.allclose(v_uncond, vf(t, x, cond), atol=1e-6)  # differs from the conditional


def test_force_uncond_mask_value_runs_and_differs():
    torch.manual_seed(0)
    vf = _conditional_vf("concatenation", condition_dropout_prob=0.1, condition_null="mask_value")
    t, x, cond = torch.rand(4, 1), torch.randn(4, 10), {"drug": torch.randn(4, 1, 3)}
    v_uncond = vf(t, x, cond, force_uncond=True)
    # mask_value null == encoding a mask_value-filled condition.
    v_masked = vf(t, x, vf.null_condition_dict(cond))
    assert torch.allclose(v_uncond, v_masked, atol=1e-6)
    assert not torch.allclose(v_uncond, vf(t, x, cond), atol=1e-6)


def test_cfg_enabled_reflects_dropout_prob():
    assert _conditional_vf("concatenation", condition_dropout_prob=0.1).cfg_enabled is True
    assert _conditional_vf("concatenation", condition_dropout_prob=0.0).cfg_enabled is False


# --- inference guidance blend ---------------------------------------------------------------------


def test_guidance_scale_one_is_plain_conditional_and_two_differs():
    torch.manual_seed(0)
    vf = _conditional_vf("concatenation", condition_dropout_prob=0.1)
    src, cond = torch.randn(6, 10), {"drug": torch.randn(6, 1, 3)}
    kw = dict(is_genot=False, state_dim=10, num_steps=5)
    y1 = integrate_translation(vf, src, cond, guidance_scale=1.0, **kw)
    y1b = integrate_translation(vf, src, cond, guidance_scale=1.0, **kw)
    y2 = integrate_translation(vf, src, cond, guidance_scale=2.0, **kw)
    assert torch.allclose(y1, y1b)  # deterministic, no-guidance path is stable
    assert not torch.allclose(y1, y2)  # w=2 actually blends in the (cond - uncond) delta


# --- training-time condition dropout --------------------------------------------------------------


def _otfm_objective(seed=0):
    from sc_flow.flow.probability_paths._probability_paths import LinearDiracProbabilityPath
    from sc_flow.training import build_objective

    return build_objective("otfm", LinearDiracProbabilityPath(sigma=0.0), match_method="independent", seed=seed)


def test_condition_dropout_training_step_is_finite_and_reproducible():
    torch.manual_seed(0)
    vf = _conditional_vf("concatenation", condition_dropout_prob=0.5, condition_null="zero_embedding")
    batch = {
        "source": torch.randn(8, 10), "target": torch.randn(8, 10),
        "condition": {"drug": torch.randn(8, 1, 3)},
    }
    loss_a, logs = _otfm_objective(seed=0).compute_loss(vf, batch)
    loss_b, _ = _otfm_objective(seed=0).compute_loss(vf, batch)
    assert torch.isfinite(loss_a)
    assert torch.allclose(loss_a, loss_b)  # seeded dropout draw => reproducible


# --- validation callback guidance sweep -----------------------------------------------------------


class _FakeMetric(torch.nn.Module):
    """Minimal torchmetrics-like stub: records mean L2 distance between pred and target populations."""

    def __init__(self):
        super().__init__()
        self._vals: list[float] = []

    def clone(self):
        return _FakeMetric()

    def update(self, pred, target):
        self._vals.append(float((pred.mean(0) - target.mean(0)).pow(2).mean()))

    def compute(self):
        return torch.tensor(sum(self._vals) / len(self._vals))

    def reset(self):
        self._vals = []


class _FakePredictor:
    """Predict = shift the source a ``w/4`` fraction toward the target mean, so across w in {1,2} a bigger
    w lands strictly closer (monotonic) — the winner is unambiguous."""

    def __init__(self, w):
        self._w = w

    def predict(self, model, batch):
        return batch["source"] + (self._w / 4.0) * (batch["target"].mean(0, keepdim=True) - batch["source"])


class _StubModule:
    def __init__(self):
        self.model = None
        self.logged: dict[str, float] = {}

    def log(self, name, value, prog_bar=False):
        self.logged[name] = float(value)


def test_validation_callback_sweeps_scales_and_selects_best():
    # lower _FakeMetric (dist) is better; a bigger w pulls the prediction closer => best_w should be 2.0.
    cb = PerturbationValidationCallback(
        predictor=_FakePredictor(1.0),
        val_metrics={"dist": _FakeMetric()},
        val_max_source_cells=None,
        guidance_predictors={1.0: _FakePredictor(1.0), 2.0: _FakePredictor(2.0)},
    )
    module = _StubModule()
    batch = {"source": torch.randn(16, 5), "target": torch.randn(16, 5) + 3.0}

    class _Trainer:
        sanity_checking = False

    cb.on_validation_batch_end(_Trainer(), module, None, batch, 0)
    cb.on_validation_epoch_end(_Trainer(), module)

    # every scale logged, plus identity, plus the un-suffixed best + the chosen scale.
    assert "val_dist_mean_gs1" in module.logged and "val_dist_mean_gs2" in module.logged
    assert "val_dist_mean_identity" in module.logged
    assert "val_dist_mean" in module.logged and "val_best_guidance_scale" in module.logged
    assert module.logged["val_best_guidance_scale"] == 2.0  # w=2 is closer => lower dist => selected
    assert cb.metrics_history["dist"][-1] == cb.metrics_history["dist_gs2"][-1]  # best == the w=2 stream


def test_validation_callback_no_sweep_is_legacy_model_stream():
    cb = PerturbationValidationCallback(
        predictor=_FakePredictor(1.0), val_metrics={"dist": _FakeMetric()}, val_max_source_cells=None,
    )
    module = _StubModule()
    batch = {"source": torch.randn(8, 5), "target": torch.randn(8, 5)}

    class _Trainer:
        sanity_checking = False

    cb.on_validation_batch_end(_Trainer(), module, None, batch, 0)
    cb.on_validation_epoch_end(_Trainer(), module)
    assert "val_dist_mean" in module.logged and "val_dist_mean_identity" in module.logged
    assert "val_best_guidance_scale" not in module.logged  # no sweep => no best-w bookkeeping
