from typing import get_args

import pytest
import torch

from sckitflow._types import ConditioningLayersId
from sckitflow.core.nn._conditioning_layers import (
    _CONDITIONING_LAYERS,
    AdaLNZeroConditioning,
    ConcatConditioning,
    FiLMConditioning,
    Resnet1dConditioning,
    get_conditioning_layer,
)

B, S, F, T, C = 8, 50, 16, 6, 8

# (time, condition, state) leading shapes: a single time or condition broadcasts over the batch and samples
CASES = [
    ((), (), (B,)),
    ((1,), (1,), (B,)),
    ((B,), (B,), (B,)),
    ((), (), (B, S)),
    ((1,), (1,), (B, S)),
    ((B,), (B,), (B, S)),
    ((B, S), (B, S), (B, S)),
]
LAYERS = [ConcatConditioning, Resnet1dConditioning, FiLMConditioning, AdaLNZeroConditioning]


@pytest.mark.parametrize("cls", LAYERS)
@pytest.mark.parametrize("with_cond", [True, False])
@pytest.mark.parametrize(("t_lead", "c_lead", "x_lead"), CASES)
def test_output_shape(cls, with_cond, t_lead, c_lead, x_lead):
    layer = cls(F, T, C if with_cond else None)
    cond = torch.zeros((*c_lead, C)) if with_cond else None
    out = layer(torch.zeros((*t_lead, T)), torch.zeros((*x_lead, F)), cond)
    assert out.shape == (*x_lead, layer.output_dim)


def test_concat_output_dim():
    assert ConcatConditioning(F, T, C).output_dim == F + T + C
    assert ConcatConditioning(F, T).output_dim == F + T


@pytest.mark.parametrize("cls", LAYERS)
def test_missing_condition_raises(cls):
    with pytest.raises(RuntimeError, match="should be passed"):
        cls(F, T, C)(torch.zeros(T), torch.zeros(B, F))


def test_film_starts_as_identity_and_depends_on_condition():
    """``(1 + gamma) * x + beta`` with a zero projection is the identity; once trained, the condition scales the state."""
    layer = FiLMConditioning(F, T, C)
    t, x = torch.randn(B, T), torch.randn(B, F)
    torch.testing.assert_close(layer(t, x, torch.randn(B, C)), x)

    torch.nn.init.normal_(layer._linear.weight)
    assert not torch.allclose(layer(t, x, torch.zeros(B, C)), layer(t, x, torch.ones(B, C)))


def test_film_modulation_is_unbounded():
    """No activation on (gamma, beta): a negative projection flips the state's sign."""
    layer = FiLMConditioning(F, T)
    torch.nn.init.zeros_(layer._linear.weight)
    with torch.no_grad():
        layer._linear.bias.copy_(torch.cat([torch.full((F,), -3.0), torch.zeros(F)]))
    x = torch.randn(B, F)
    torch.testing.assert_close(layer(torch.zeros(T), x), -2 * x)


def test_adaln_zero_starts_as_layer_norm():
    layer = AdaLNZeroConditioning(F, T, C)
    x = torch.randn(B, S, F) * 5 + 3
    out = layer(torch.randn(B, T), x, torch.randn(B, C))
    torch.testing.assert_close(out, torch.nn.functional.layer_norm(x, (F,)))


def test_registry_matches_the_literal():
    assert set(_CONDITIONING_LAYERS) == set(get_args(ConditioningLayersId))


@pytest.mark.parametrize("conditioning_id", list(_CONDITIONING_LAYERS))
def test_get_conditioning_layer(conditioning_id):
    layer = get_conditioning_layer(F, T, latent_condition_dim=C, conditioning_id=conditioning_id)
    assert isinstance(layer, _CONDITIONING_LAYERS[conditioning_id])


def test_get_conditioning_layer_unknown_raises():
    with pytest.raises(ValueError, match="not available"):
        get_conditioning_layer(F, T, conditioning_id="nope")
