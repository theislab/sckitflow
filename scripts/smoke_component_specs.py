"""Standalone verification for the component-spec design (registries, configs, portable bundle).

Not a pytest test (the pytest suite is currently stale — see ``docs/plans/state.md`` §14). Run directly:

    python scripts/smoke_component_specs.py

Exits non-zero on the first failed check. Covers: the generic ComponentRegistry validation contract, the
pooling + combiner discriminated specs, the activation enum, the MLPVelocityConfig/SetEncoderConfig JSON
round-trip, and the portable ``save_pretrained`` / ``from_pretrained`` bundle (including the "fail before
writing" guard for runtime-only custom modules).
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import torch

from sc_flow.core._component import ComponentRegistry
from sc_flow.flow import MLPVelocity, MLPVelocityConfig, SetEncoder
from sc_flow.flow._combiner import build_combiner, validate_combiner_spec
from sc_flow.flow._config import MLPEmbedderConfig
from sc_flow.flow._pooling import MeanPooling, build_pooling, validate_pooling_spec

TOKEN = {
    "type": "sc_flow.attention_token",
    "version": 1,
    "config": {
        "num_heads": 4,
        "qkv_dim": 16,
        "dropout": 0.0,
        "num_layers": 1,
        "transformer_block": False,
        "layer_norm": False,
        "ff_dim": None,
    },
}
CONCAT = {"type": "sc_flow.concat", "version": 1, "config": {}}
RESNET = {"type": "sc_flow.resnet1d", "version": 1, "config": {"num_resnet_layers": 2}}


def check(name: str, cond: bool) -> None:
    """Assert ``cond`` holds for the named check, printing an ok line."""
    if not cond:
        raise AssertionError(f"FAILED: {name}")
    print(f"  ok: {name}")


def expect_error(name: str, fn) -> None:
    """Assert calling ``fn`` raises a ValueError/TypeError for the named check."""
    try:
        fn()
    except (ValueError, TypeError):
        print(f"  ok: {name} (raised)")
        return
    raise AssertionError(f"FAILED: {name} did not raise")


def test_generic_registry() -> None:
    """The generic ComponentRegistry validation contract (via the pooling registry)."""
    print("[generic registry]")
    reg = validate_pooling_spec  # the pooling registry is a ComponentRegistry instance under the hood
    check("canonicalizes token config", reg(TOKEN)["config"]["num_heads"] == 4)
    expect_error("unknown field", lambda: reg({**TOKEN, "extra": 1}))
    expect_error("missing field", lambda: reg({"type": "sc_flow.mean", "version": 1}))
    expect_error("unknown type", lambda: reg({"type": "nope", "version": 1, "config": {}}))
    expect_error("bad version", lambda: reg({"type": "sc_flow.mean", "version": 2, "config": {}}))
    expect_error("non-json config", lambda: reg({"type": "sc_flow.mean", "version": 1, "config": {"x": object()}}))
    expect_error("wrong config field", lambda: reg({"type": "sc_flow.mean", "version": 1, "config": {"x": 1}}))
    # duplicate-registration guard: idempotent for the same binding, errors on a different one.
    from sc_flow.flow._pooling import MeanPoolingConfig

    r = ComponentRegistry("demo")
    build_fn = lambda c, ctx: None
    r.register("t", config_type=MeanPoolingConfig, build=build_fn)
    r.register("t", config_type=MeanPoolingConfig, build=build_fn)  # same binding -> no error
    check("idempotent re-registration", r.types == ["t"])
    expect_error(
        "re-register different binding",
        lambda: r.register("t", config_type=MeanPoolingConfig, build=lambda c, ctx: 1),
    )


def test_pooling() -> None:
    """Pooling specs build; permutation invariance, masking correctness, all-masked error."""
    print("[pooling]")
    x = torch.randn(3, 5, 16)
    for spec in (
        {"type": "sc_flow.mean", "version": 1, "config": {}},
        {"type": "sc_flow.sum", "version": 1, "config": {}},
        TOKEN,
    ):
        p = build_pooling(spec, input_dim=16)
        check(f"{spec['type']} outputs (batch, feat)", p(x).shape == (3, 16))
    seed = {
        "type": "sc_flow.attention_seed",
        "version": 1,
        "config": {"num_heads": 4, "v_dim": 8, "seed_dim": 8, "dropout": 0.0,
                   "transformer_block": False, "layer_norm": False, "ff_dim": None},
    }
    check("seed attention changes output dim", build_pooling(seed, input_dim=16)(x).shape == (3, 8))

    # permutation invariance (mean)
    mean = MeanPooling(16).eval()
    perm = torch.randperm(5)
    check("mean pooling is permutation invariant", torch.allclose(mean(x), mean(x[:, perm, :]), atol=1e-6))

    # masked mean correctness + mixed-length masks in one batch
    mask = torch.ones(3, 5, dtype=torch.bool)
    mask[0, 3:] = False  # first example: only 3 valid
    mask[1, 1:] = False  # second: only 1 valid
    got = mean(x, mask)
    want0 = x[0, :3].mean(0)
    check("masked mean uses only valid tokens", torch.allclose(got[0], want0, atol=1e-6))
    check("padded values do not affect masked mean", torch.allclose(mean(x, mask), mean(x * mask.unsqueeze(-1) + torch.randn_like(x) * (~mask).unsqueeze(-1), mask), atol=1e-5))
    # all-masked example is undefined -> raises
    bad = torch.ones(3, 5, dtype=torch.bool)
    bad[2] = False
    expect_error("all-masked example raises", lambda: mean(x, bad))


def test_combiner() -> None:
    """Combiner specs build and canonicalize; unknown type is rejected."""
    print("[combiner]")
    check("resnet spec canonicalizes", validate_combiner_spec(RESNET)["type"] == "sc_flow.resnet1d")
    c = build_combiner(CONCAT, latent_state_dim=8, latent_time_dim=4, latent_condition_dim=6)
    et, es, ec = torch.randn(2, 4), torch.randn(2, 8), torch.randn(2, 6)
    check("concat output dim is sum", c(et, es, encoded_condition=ec).shape == (2, 18))
    expect_error("unknown combiner type", lambda: validate_combiner_spec({"type": "sc_flow.nope", "version": 1, "config": {}}))


def test_activation() -> None:
    """The activation enum resolves ids/classes and rejects non-portable classes."""
    print("[activation]")
    from sc_flow.core.nn._activation import activation_id, resolve_activation

    check("id resolves to class", resolve_activation("tanh", "relu") is torch.nn.Tanh)
    check("None resolves to default", resolve_activation(None, "silu") is torch.nn.SiLU)
    check("class maps back to id", activation_id(torch.nn.Tanh, "relu") == "tanh")
    expect_error("unknown id", lambda: resolve_activation("swish", "relu"))
    expect_error("non-portable class has no id", lambda: activation_id(torch.nn.Mish, "relu"))


def _build_conditional_vf(combiner, mode="stochastic") -> MLPVelocity:
    """A conditional MLPVelocity used across the round-trip and bundle checks."""
    return MLPVelocity(
        state_dim=12,
        combiner=combiner,
        state_embedder=MLPEmbedderConfig(output_dim=32, mlp_kwargs={"hidden_dims": [16], "activation_cls": "tanh"}),
        time_embedder=MLPEmbedderConfig(output_dim=16),
        time_features_id="sinusoidal",
        num_time_features=8,
        condition_encoder=SetEncoder(
            input_layers={"drug": {"input_dim": 5, "output_dim": 16}, "cl": {"input_dim": 3, "output_dim": 16}},
            output_dim=32,
            pooling=TOKEN,
            condition_mode=mode,
        ),
    )


def test_config_roundtrip() -> None:
    """MLPVelocityConfig round-trips through JSON with identical forward output."""
    print("[config round-trip]")
    t, x = torch.rand(4), torch.randn(4, 12)
    cond = {"drug": torch.randn(4, 3, 5), "cl": torch.randn(4, 2, 3)}
    for combiner in (CONCAT, RESNET):
        vf = _build_conditional_vf(combiner)
        cfg = vf.to_config()
        rebuilt = MLPVelocityConfig.from_dict(json.loads(json.dumps(cfg.to_dict())))
        vf2 = MLPVelocity.from_config(rebuilt)
        vf2.load_state_dict(vf.state_dict())
        vf.eval()
        vf2.eval()
        y1 = vf(t, x, condition_dict=cond)
        y2 = vf2(t, x, condition_dict=cond)
        check(f"{combiner['type']} config JSON round-trips", cfg.to_dict() == rebuilt.to_dict())
        check(f"{combiner['type']} forward equal after rebuild+load", torch.allclose(y1, y2, atol=1e-6))


def test_bundle() -> None:
    """The portable save_pretrained/from_pretrained bundle; runtime-only slots refuse export."""
    print("[portable bundle]")
    t, x = torch.rand(4), torch.randn(4, 12)
    cond = {"drug": torch.randn(4, 3, 5), "cl": torch.randn(4, 2, 3)}
    vf = _build_conditional_vf(CONCAT)
    vf.eval()
    y0 = vf(t, x, condition_dict=cond)
    with tempfile.TemporaryDirectory() as d:
        vf.save_pretrained(d)
        files = {p.name for p in Path(d).iterdir()}
        check("bundle is config.json + model.safetensors", files == {"config.json", "model.safetensors"})
        vf2 = MLPVelocity.from_pretrained(d)
        vf2.eval()
        check("conditional VF bundle round-trips exactly", torch.allclose(y0, vf2(t, x, condition_dict=cond), atol=1e-6))

    # runtime-only custom modules refuse portable export (before writing).
    se_custom = SetEncoder(
        input_layers={"drug": {"input_dim": 5, "output_dim": 16}}, output_dim=16, pooling=MeanPooling(16)
    )
    vf_custom_pool = MLPVelocity(
        state_dim=12, combiner=CONCAT, state_embedder=MLPEmbedderConfig(output_dim=16),
        time_embedder=MLPEmbedderConfig(output_dim=8), condition_encoder=se_custom,
    )
    expect_error("custom pooling refuses export", vf_custom_pool.to_config)


def test_no_hidden_defaults() -> None:
    """Architecture dims/counts have no hidden defaults: omitting a required one raises."""
    print("[no hidden defaults]")
    # embedder output width is required
    expect_error("MLPEmbedderConfig requires output_dim", lambda: MLPEmbedderConfig())
    # num_time_features is required when a featurizer is selected
    expect_error(
        "num_time_features required when featurizing",
        lambda: MLPVelocity(
            state_dim=8, combiner=CONCAT, state_embedder=MLPEmbedderConfig(output_dim=8),
            time_embedder=MLPEmbedderConfig(output_dim=8), time_features_id="sinusoidal", num_time_features=None,
        ),
    )
    # resnet combiner depth is a required, explicit config field (not a hidden default)
    expect_error(
        "resnet combiner requires num_resnet_layers",
        lambda: validate_combiner_spec({"type": "sc_flow.resnet1d", "version": 1, "config": {}}),
    )


def main() -> int:
    """Run every check group; return 0 on success."""
    groups = (
        test_generic_registry,
        test_pooling,
        test_combiner,
        test_activation,
        test_config_roundtrip,
        test_bundle,
        test_no_hidden_defaults,
    )
    for fn in groups:
        fn()
    print("\nPASS: component-spec design verified end-to-end.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as e:
        print(f"\n{e}", file=sys.stderr)
        sys.exit(1)
