"""Tests for :class:`FiLMConditioning` and :class:`FiLMConditioningConfig`."""

from __future__ import annotations

import pytest
import torch

from sckitflow.core.nn.blocks.conditioning._base import BaseConditioningLayerConfig
from sckitflow.core.nn.blocks.conditioning._film import (
    FiLMConditioning,
    FiLMConditioningConfig,
)

_BATCH = 4
_SAMPLES = 5
_STATE = 16
_TIME = 6
_COND = 8

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


def _make_layer(*, with_cond: bool) -> FiLMConditioning:
    """``FiLMConditioning.__init__`` is keyword-only — the historical test
    passed positional args and would now raise ``TypeError``."""
    return FiLMConditioning(
        latent_state_dim=_STATE,
        latent_time_dim=_TIME,
        latent_condition_dim=_COND if with_cond else None,
    )


class TestFiLMConditioning:
    # ---- forward shape matrix -------------------------------------------

    @pytest.mark.parametrize("x_shape,t_shape,cond_shape", SHAPE_CASES)
    @pytest.mark.parametrize("with_cond", [True, False])
    def test_forward_shape(self, x_shape, t_shape, cond_shape, with_cond) -> None:
        layer = _make_layer(with_cond=with_cond)
        if not with_cond:
            cond_shape = None
        x, t, cond = _input_tensors(x_shape, t_shape, cond_shape)
        out = layer(t, x, cond) if cond is not None else layer(t, x)
        # FiLM preserves the state shape verbatim.
        assert out.shape == x_shape
        assert out.shape[-1] == layer.output_dim

    # ---- dims -----------------------------------------------------------

    def test_output_dim_is_state_dim(self) -> None:
        assert _make_layer(with_cond=False).output_dim == _STATE
        assert _make_layer(with_cond=True).output_dim == _STATE

    def test_embedding_dim_no_cond(self) -> None:
        assert _make_layer(with_cond=False).embedding_dim == _TIME

    def test_embedding_dim_with_cond(self) -> None:
        assert _make_layer(with_cond=True).embedding_dim == _TIME + _COND

    # ---- semantics: output equals gamma * x + beta ---------------------

    def test_output_is_gamma_x_plus_beta(self) -> None:
        """Recompute the documented formula manually from the module's own
        linear + activation and compare against ``forward``.
        """
        torch.manual_seed(0)
        layer = _make_layer(with_cond=False)
        x = torch.randn(_BATCH, _STATE)
        t = torch.randn(_BATCH, _TIME)
        with torch.no_grad():
            gamma, beta = layer._activation(layer._linear(t)).chunk(2, dim=-1)
            expected = gamma * x + beta
            actual = layer(t, x)
        assert torch.allclose(actual, expected)

    def test_zeroed_linear_gives_zero_output(self) -> None:
        """``SiLU(0) = 0`` ⇒ ``gamma = beta = 0`` ⇒ ``out = 0 * x + 0 = 0``."""
        layer = _make_layer(with_cond=False)
        with torch.no_grad():
            layer._linear.weight.zero_()
            layer._linear.bias.zero_()
        x = torch.randn(_BATCH, _STATE)
        t = torch.randn(_BATCH, _TIME)
        assert torch.allclose(layer(t, x), torch.zeros_like(x))

    def test_output_depends_on_time(self) -> None:
        """Guard against a future refactor that drops the modulation."""
        torch.manual_seed(0)
        layer = _make_layer(with_cond=False)
        x = torch.randn(_BATCH, _STATE)
        t1 = torch.randn(_BATCH, _TIME)
        t2 = torch.randn(_BATCH, _TIME)
        assert not torch.allclose(layer(t1, x), layer(t2, x))

    def test_output_depends_on_condition(self) -> None:
        torch.manual_seed(0)
        layer = _make_layer(with_cond=True)
        x = torch.randn(_BATCH, _STATE)
        t = torch.randn(_BATCH, _TIME)
        c1 = torch.randn(_BATCH, _COND)
        c2 = torch.randn(_BATCH, _COND)
        assert not torch.allclose(layer(t, x, c1), layer(t, x, c2))

    # ---- validation is wired up -----------------------------------------

    def test_validates_state_shape(self) -> None:
        layer = _make_layer(with_cond=False)
        with pytest.raises(RuntimeError, match="state"):
            layer(torch.zeros(_BATCH, _TIME), torch.zeros(_BATCH, _STATE + 1))

    # ---- gradients ------------------------------------------------------

    def test_gradients_flow(self) -> None:
        layer = _make_layer(with_cond=False)
        x = torch.randn(_BATCH, _STATE, requires_grad=True)
        t = torch.randn(_BATCH, _TIME)
        layer(t, x).sum().backward()
        assert x.grad is not None
        assert layer._linear.weight.grad is not None


class TestFiLMConditioningConfig:
    def test_extends_base(self) -> None:
        assert issubclass(FiLMConditioningConfig, BaseConditioningLayerConfig)

    def test_build_returns_film(self) -> None:
        layer = FiLMConditioningConfig().build(latent_state_dim=_STATE, latent_time_dim=_TIME)
        assert isinstance(layer, FiLMConditioning)
        assert layer.output_dim == _STATE

    def test_build_with_cond(self) -> None:
        layer = FiLMConditioningConfig().build(
            latent_state_dim=_STATE,
            latent_time_dim=_TIME,
            latent_condition_dim=_COND,
        )
        assert layer.embedding_dim == _TIME + _COND

    def test_build_forward_roundtrip(self) -> None:
        layer = FiLMConditioningConfig().build(latent_state_dim=_STATE, latent_time_dim=_TIME)
        x = torch.randn(_BATCH, _STATE)
        t = torch.randn(_BATCH, _TIME)
        assert layer(t, x).shape == (_BATCH, _STATE)
