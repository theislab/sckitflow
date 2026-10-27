"""Tests for :class:`ConcatConditioning` and :class:`ConcatConditioningConfig`."""

from __future__ import annotations

import pytest
import torch

from sckitflow.core.nn.blocks.conditioning._base import BaseConditioningLayerConfig
from sckitflow.core.nn.blocks.conditioning._concat import (
    ConcatConditioning,
    ConcatConditioningConfig,
)

_BATCH = 4
_SAMPLES = 5
_STATE = 16
_TIME = 6
_COND = 8

# The seven (x, t, cond) shape combinations the historical test enumerated:
# unbatched / batched / extra leading dims, plus every broadcast of t and cond.
SHAPE_CASES = [
    pytest.param((_BATCH, _STATE), (_TIME,), (_COND,), id="b-s|t|c"),
    pytest.param((_BATCH, _STATE), (1, _TIME), (1, _COND), id="b-s|1-t|1-c"),
    pytest.param((_BATCH, _STATE), (_BATCH, _TIME), (_BATCH, _COND), id="b-s|b-t|b-c"),
    pytest.param((_BATCH, _SAMPLES, _STATE), (_TIME,), (_COND,), id="b-n-s|t|c"),
    pytest.param((_BATCH, _SAMPLES, _STATE), (1, _TIME), (1, _COND), id="b-n-s|1-t|1-c"),
    pytest.param((_BATCH, _SAMPLES, _STATE), (_BATCH, _TIME), (_BATCH, _COND), id="b-n-s|b-t|b-c"),
    pytest.param(
        (_BATCH, _SAMPLES, _STATE),
        (_BATCH, _SAMPLES, _TIME),
        (_BATCH, _SAMPLES, _COND),
        id="b-n-s|b-n-t|b-n-c",
    ),
]


def _input_tensors(x_shape, t_shape, cond_shape):
    x = torch.randn(*x_shape)
    t = torch.randn(*t_shape)
    cond = torch.randn(*cond_shape) if cond_shape is not None else None
    return x, t, cond


class TestConcatConditioning:
    # ---- forward shape matrix -------------------------------------------

    @pytest.mark.parametrize("x_shape,t_shape,cond_shape", SHAPE_CASES)
    @pytest.mark.parametrize("with_cond", [True, False])
    def test_forward_shape(self, x_shape, t_shape, cond_shape, with_cond) -> None:
        if with_cond:
            layer = ConcatConditioning(_STATE, _TIME, _COND)
            expected_dim = _STATE + _TIME + _COND
        else:
            layer = ConcatConditioning(_STATE, _TIME)
            expected_dim = _STATE + _TIME
            cond_shape = None

        x, t, cond = _input_tensors(x_shape, t_shape, cond_shape)
        out = layer(t, x, cond) if cond is not None else layer(t, x)
        assert out.shape == x_shape[:-1] + (expected_dim,)
        assert out.shape[-1] == layer.output_dim

    # ---- output_dim -----------------------------------------------------

    def test_output_dim_no_cond(self) -> None:
        assert ConcatConditioning(_STATE, _TIME).output_dim == _STATE + _TIME

    def test_output_dim_with_cond(self) -> None:
        layer = ConcatConditioning(_STATE, _TIME, _COND)
        assert layer.output_dim == _STATE + _TIME + _COND

    # ---- semantics: output is literally the concatenation --------------

    def test_output_is_concat_in_order(self) -> None:
        layer = ConcatConditioning(_STATE, _TIME, _COND)
        x = torch.randn(_BATCH, _STATE)
        t = torch.randn(_BATCH, _TIME)
        cond = torch.randn(_BATCH, _COND)
        out = layer(t, x, cond)
        assert torch.allclose(out[:, :_STATE], x)
        assert torch.allclose(out[:, _STATE : _STATE + _TIME], t)
        assert torch.allclose(out[:, _STATE + _TIME :], cond)

    def test_output_is_concat_in_order_no_cond(self) -> None:
        layer = ConcatConditioning(_STATE, _TIME)
        x = torch.randn(_BATCH, _STATE)
        t = torch.randn(_BATCH, _TIME)
        out = layer(t, x)
        assert torch.allclose(out[:, :_STATE], x)
        assert torch.allclose(out[:, _STATE:], t)

    def test_output_with_broadcast(self) -> None:
        layer = ConcatConditioning(_STATE, _TIME, _COND)
        x = torch.randn(_BATCH, _STATE)
        t = torch.randn(_TIME)
        cond = torch.randn(_COND)
        out = layer(t, x, cond)
        assert torch.allclose(out[:, :_STATE], x)
        assert torch.allclose(out[:, _STATE : _STATE + _TIME], t.expand(_BATCH, _TIME))
        assert torch.allclose(out[:, _STATE + _TIME :], cond.expand(_BATCH, _COND))

    # ---- validation is wired up -----------------------------------------

    def test_validates_state_shape(self) -> None:
        layer = ConcatConditioning(_STATE, _TIME)
        with pytest.raises(RuntimeError, match="state"):
            layer(torch.zeros(_BATCH, _TIME), torch.zeros(_BATCH, _STATE + 1))

    # ---- gradients ------------------------------------------------------

    def test_gradients_flow(self) -> None:
        layer = ConcatConditioning(_STATE, _TIME)
        x = torch.randn(_BATCH, _STATE, requires_grad=True)
        t = torch.randn(_BATCH, _TIME)
        layer(t, x).sum().backward()
        assert x.grad is not None


class TestConcatConditioningConfig:
    def test_extends_base(self) -> None:
        assert issubclass(ConcatConditioningConfig, BaseConditioningLayerConfig)

    def test_build_returns_concat(self) -> None:
        layer = ConcatConditioningConfig().build(latent_state_dim=_STATE, latent_time_dim=_TIME)
        assert isinstance(layer, ConcatConditioning)
        assert layer.output_dim == _STATE + _TIME

    def test_build_with_cond(self) -> None:
        layer = ConcatConditioningConfig().build(
            latent_state_dim=_STATE,
            latent_time_dim=_TIME,
            latent_condition_dim=_COND,
        )
        assert layer.output_dim == _STATE + _TIME + _COND

    def test_build_forward_roundtrip(self) -> None:
        layer = ConcatConditioningConfig().build(latent_state_dim=_STATE, latent_time_dim=_TIME)
        x = torch.randn(_BATCH, _STATE)
        t = torch.randn(_BATCH, _TIME)
        assert layer(t, x).shape == (_BATCH, _STATE + _TIME)
