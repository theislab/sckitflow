import pytest
import torch
from torch.nn import functional as F

from sc_flow.flow._objectives import _condition_masks
from sc_flow.flow._pooling import (
    MeanPooling,
    PoolingSpec,
    SeedAttentionPooling,
    SumPooling,
    TokenAttentionPooling,
    build_pooling,
    validate_pooling_spec,
)

TOKEN_CONFIG = {
    "num_heads": 8,
    "qkv_dim": 64,
    "dropout": 0.0,
    "num_layers": 1,
    "transformer_block": False,
    "layer_norm": False,
    "ff_dim": None,
}
SEED_CONFIG = {
    "num_heads": 8,
    "v_dim": 64,
    "seed_dim": 64,
    "dropout": 0.0,
    "transformer_block": False,
    "layer_norm": False,
    "ff_dim": None,
}


def explicit_spec(type_id: str, **changes) -> PoolingSpec:
    configs = {
        "sc_flow.mean": {},
        "sc_flow.sum": {},
        "sc_flow.attention_token": TOKEN_CONFIG,
        "sc_flow.attention_seed": SEED_CONFIG,
    }
    config = {**configs[type_id], **changes}
    return PoolingSpec(type=type_id, version=1, config=config)


def test_attention_token_spec_preserves_explicit_config() -> None:
    spec = validate_pooling_spec(explicit_spec("sc_flow.attention_token"))

    assert spec == {
        "type": "sc_flow.attention_token",
        "version": 1,
        "config": TOKEN_CONFIG,
    }


@pytest.mark.parametrize(
    ("type_id", "expected_class", "expected_dim"),
    [
        ("sc_flow.mean", MeanPooling, 7),
        ("sc_flow.sum", SumPooling, 7),
        ("sc_flow.attention_token", TokenAttentionPooling, 7),
        ("sc_flow.attention_seed", SeedAttentionPooling, 64),
    ],
)
def test_closed_registry_builds_initial_variants(type_id: str, expected_class: type, expected_dim: int) -> None:
    pooling = build_pooling(explicit_spec(type_id), input_dim=7)

    assert isinstance(pooling, expected_class)
    assert pooling.output_dim == expected_dim


@pytest.mark.parametrize(
    "type_id", ["sc_flow.mean", "sc_flow.sum", "sc_flow.attention_token", "sc_flow.attention_seed"]
)
def test_pooling_is_permutation_invariant(type_id: str) -> None:
    torch.manual_seed(0)
    pooling = build_pooling(explicit_spec(type_id), input_dim=7).eval()
    x = torch.randn(3, 5, 7)
    mask = torch.tensor(
        [
            [True, True, True, False, False],
            [True, False, True, True, False],
            [True, True, True, True, True],
        ]
    )
    permutation = torch.tensor([2, 4, 0, 3, 1])

    expected = pooling(x, mask)
    actual = pooling(x[:, permutation], mask[:, permutation])

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    "type_id", ["sc_flow.mean", "sc_flow.sum", "sc_flow.attention_token", "sc_flow.attention_seed"]
)
def test_masked_padding_values_do_not_affect_mixed_length_batch(type_id: str) -> None:
    torch.manual_seed(0)
    pooling = build_pooling(explicit_spec(type_id), input_dim=7).eval()
    x = torch.randn(3, 5, 7)
    mask = torch.tensor(
        [
            [True, False, False, False, False],
            [True, True, True, False, False],
            [True, True, True, True, True],
        ]
    )
    changed_padding = x.masked_fill(~mask.unsqueeze(-1), 1_000_000)

    expected = pooling(x, mask)
    actual = pooling(changed_padding, mask)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    "type_id", ["sc_flow.mean", "sc_flow.sum", "sc_flow.attention_token", "sc_flow.attention_seed"]
)
def test_all_masked_example_is_rejected(type_id: str) -> None:
    pooling = build_pooling(explicit_spec(type_id), input_dim=7)
    x = torch.randn(2, 3, 7)
    mask = torch.tensor([[True, False, False], [False, False, False]])

    with pytest.raises(ValueError, match=r"all-masked.*\[1\]"):
        pooling(x, mask)


def test_attention_config_validation() -> None:
    with pytest.raises(ValueError, match="must be divisible"):
        validate_pooling_spec(explicit_spec("sc_flow.attention_token", num_heads=3, qkv_dim=8))
    with pytest.raises(ValueError, match="only apply when transformer_block=True"):
        validate_pooling_spec(explicit_spec("sc_flow.attention_token", layer_norm=True))
    with pytest.raises(ValueError, match="Unknown pooling type"):
        validate_pooling_spec({"type": "other.attention", "version": 1, "config": {}})


def test_attention_config_rejects_omitted_fields_instead_of_filling_defaults() -> None:
    with pytest.raises(ValueError, match="Invalid config.*dropout"):
        validate_pooling_spec(
            {
                "type": "sc_flow.attention_token",
                "version": 1,
                "config": {"num_heads": 8, "qkv_dim": 64},
            }
        )
    with pytest.raises(ValueError, match="Missing pooling spec field"):
        validate_pooling_spec({"type": "sc_flow.mean", "config": {}})
    with pytest.raises(TypeError, match="Expected a pooling spec mapping"):
        validate_pooling_spec("mean")


def test_model_apis_require_an_explicit_pooling_choice() -> None:
    from dataclasses import MISSING, fields

    from sc_flow._model import FlowMatchingConfig
    from sc_flow.flow._config import SetEncoderConfig

    # Pooling is chosen in the recipe/config (a PoolingSpec), not a model-constructor kwarg. The "no silent
    # default" contract now lives on these config dataclasses: the ``pooling`` field is required (no default).
    for config_cls in (FlowMatchingConfig, SetEncoderConfig):
        (pooling_field,) = (f for f in fields(config_cls) if f.name == "pooling")
        assert pooling_field.default is MISSING
        assert pooling_field.default_factory is MISSING


@pytest.mark.parametrize("type_id", ["sc_flow.attention_token", "sc_flow.attention_seed"])
def test_attention_dropout_is_deterministic_in_eval(type_id: str) -> None:
    spec = explicit_spec(type_id, dropout=0.75)
    pooling = build_pooling(spec, input_dim=7).eval()
    x = torch.randn(2, 4, 7)

    first = pooling(x)
    second = pooling(x)

    torch.testing.assert_close(second, first)


@pytest.mark.parametrize(
    "type_id",
    ["sc_flow.mean", "sc_flow.sum", "sc_flow.attention_token", "sc_flow.attention_seed"],
)
def test_dense_fast_path_does_not_allocate_boolean_masks(monkeypatch, type_id: str) -> None:
    pooling = build_pooling(explicit_spec(type_id), input_dim=7).eval()
    x = torch.randn(2, 4, 7)

    def fail_mask_allocation(*args, **kwargs):
        raise AssertionError("The dense no-mask path must not allocate an all-ones mask.")

    monkeypatch.setattr(torch, "ones", fail_mask_allocation)
    output = pooling(x)

    assert output.shape == (2, pooling.output_dim)


def test_dense_attention_fast_path_calls_sdpa_without_attention_mask(monkeypatch) -> None:
    pooling = build_pooling(explicit_spec("sc_flow.attention_token"), input_dim=7).eval()
    original_sdpa = F.scaled_dot_product_attention
    observed_masks = []

    def record_mask(query, key, value, *, attn_mask=None, dropout_p=0.0):
        observed_masks.append(attn_mask)
        return original_sdpa(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p)

    monkeypatch.setattr(F, "scaled_dot_product_attention", record_mask)
    pooling(torch.randn(2, 4, 7))

    assert observed_masks == [None]


def test_condition_masks_follow_target_coupling_reorder() -> None:
    mask = {"drug": torch.tensor([[True, False], [True, True], [False, True]])}
    target_indices = torch.tensor([2, 0])

    reordered = _condition_masks(mask, target_indices, n_target=3, device="cpu")

    torch.testing.assert_close(reordered["drug"], mask["drug"][target_indices])
