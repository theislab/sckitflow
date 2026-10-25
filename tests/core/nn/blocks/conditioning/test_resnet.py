"""Tests for :class:`ResNetConditioning` and :class:`ResNetConditioningConfig`.

The source module currently has a ``SyntaxError`` (bare ``*`` without a
trailing comma in ``ResNetConditioning.__init__``) which prevents the whole
``sckitflow.core.nn.conditioning._resnet`` module from being imported. The
lazy import guard below ensures the test file still collects, and the
``test_resnet_conditioning_module_importable`` diagnostic fails loudly with
the underlying exception so a CI run does not silently skip everything.
"""

from __future__ import annotations

import pytest
import torch

from sckitflow.core.nn.blocks._resnet import ResNetConfig
from sckitflow.core.nn.blocks.conditioning._base import BaseConditioningLayerConfig
from sckitflow.core.nn.blocks.conditioning._resnet import (
    ResNetConditioning,
    ResNetConditioningConfig,
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


def _small_cfg() -> ResNetConfig:
    """One residual layer, so tests stay fast."""
    return ResNetConfig(num_resnet_layers=1)


# ---------------------------------------------------------------------------
# Diagnostic — runs unconditionally so import breakage is always visible.
# ---------------------------------------------------------------------------


class TestResNetConditioning:
    # ---- forward shape matrix -------------------------------------------

    @pytest.mark.parametrize("x_shape,t_shape,cond_shape", SHAPE_CASES)
    @pytest.mark.parametrize("with_cond", [True, False])
    def test_forward_shape(self, x_shape, t_shape, cond_shape, with_cond) -> None:
        if with_cond:
            layer = ResNetConditioning(
                latent_state_dim=_STATE,
                latent_time_dim=_TIME,
                latent_condition_dim=_COND,
                resnet_cfg=_small_cfg(),
            )
        else:
            layer = ResNetConditioning(
                latent_state_dim=_STATE,
                latent_time_dim=_TIME,
                resnet_cfg=_small_cfg(),
            )
            cond_shape = None

        x, t, cond = _input_tensors(x_shape, t_shape, cond_shape)
        out = layer(t, x, cond) if cond is not None else layer(t, x)
        # ResNet preserves the state dimensionality by default.
        assert out.shape == x_shape
        assert out.shape[-1] == layer.output_dim

    # ---- dims -----------------------------------------------------------

    def test_output_dim(self) -> None:
        layer = ResNetConditioning(
            latent_state_dim=_STATE,
            latent_time_dim=_TIME,
            resnet_cfg=_small_cfg(),
        )
        assert layer.output_dim == _STATE

    def test_embedding_dim_no_cond(self) -> None:
        layer = ResNetConditioning(
            latent_state_dim=_STATE,
            latent_time_dim=_TIME,
            resnet_cfg=_small_cfg(),
        )
        assert layer.embedding_dim == _TIME

    def test_embedding_dim_with_cond(self) -> None:
        layer = ResNetConditioning(
            latent_state_dim=_STATE,
            latent_time_dim=_TIME,
            latent_condition_dim=_COND,
            resnet_cfg=_small_cfg(),
        )
        assert layer.embedding_dim == _TIME + _COND

    # ---- default cfg (regression for the `resnet_cfg` vs `self._resnet_cfg` bug) --

    def test_default_resnet_cfg(self) -> None:
        """``resnet_cfg=None`` must fall back to ``ResNetConfig()``. The
        current source reads ``resnet_cfg.build(...)`` (the parameter) rather
        than ``self._resnet_cfg.build(...)``, so this raises
        ``AttributeError: 'NoneType' object has no attribute 'build'``.
        """
        layer = ResNetConditioning(
            latent_state_dim=_STATE,
            latent_time_dim=_TIME,
        )
        assert layer.output_dim == _STATE

    # ---- semantics ------------------------------------------------------

    def test_output_depends_on_time(self) -> None:
        torch.manual_seed(0)
        layer = ResNetConditioning(
            latent_state_dim=_STATE,
            latent_time_dim=_TIME,
            resnet_cfg=_small_cfg(),
        )
        x = torch.randn(_BATCH, _STATE)
        t1 = torch.randn(_BATCH, _TIME)
        t2 = torch.randn(_BATCH, _TIME)
        assert not torch.allclose(layer(t1, x), layer(t2, x))

    def test_output_depends_on_condition(self) -> None:
        torch.manual_seed(0)
        layer = ResNetConditioning(
            latent_state_dim=_STATE,
            latent_time_dim=_TIME,
            latent_condition_dim=_COND,
            resnet_cfg=_small_cfg(),
        )
        x = torch.randn(_BATCH, _STATE)
        t = torch.randn(_BATCH, _TIME)
        c1 = torch.randn(_BATCH, _COND)
        c2 = torch.randn(_BATCH, _COND)
        assert not torch.allclose(layer(t, x, c1), layer(t, x, c2))

    # ---- validation is wired up -----------------------------------------

    def test_validates_state_shape(self) -> None:
        layer = ResNetConditioning(
            latent_state_dim=_STATE,
            latent_time_dim=_TIME,
            resnet_cfg=_small_cfg(),
        )
        with pytest.raises(RuntimeError, match="state"):
            layer(torch.zeros(_BATCH, _TIME), torch.zeros(_BATCH, _STATE + 1))

    # ---- gradients ------------------------------------------------------

    def test_gradients_flow(self) -> None:
        layer = ResNetConditioning(
            latent_state_dim=_STATE,
            latent_time_dim=_TIME,
            resnet_cfg=_small_cfg(),
        )
        x = torch.randn(_BATCH, _STATE, requires_grad=True)
        t = torch.randn(_BATCH, _TIME)
        layer(t, x).sum().backward()
        assert x.grad is not None


class TestResNetConditioningConfig:
    def test_extends_base(self) -> None:
        assert issubclass(ResNetConditioningConfig, BaseConditioningLayerConfig)

    def test_build_returns_resnet_conditioning(self) -> None:
        """Current source: ``build`` returns ``Resnet1dConditioning`` (a
        ``NameError``), and is annotated as ``-> ResNet``. Both are wrong;
        this test fails with ``NameError`` on the current source.
        """
        cfg = ResNetConditioningConfig(resnet=_small_cfg())
        layer = cfg.build(latent_state_dim=_STATE, latent_time_dim=_TIME)
        assert isinstance(layer, ResNetConditioning)
        assert layer.output_dim == _STATE

    def test_build_with_cond(self) -> None:
        cfg = ResNetConditioningConfig(resnet=_small_cfg())
        layer = cfg.build(
            latent_state_dim=_STATE,
            latent_time_dim=_TIME,
            latent_condition_dim=_COND,
        )
        assert layer.embedding_dim == _TIME + _COND

    def test_build_forward_roundtrip(self) -> None:
        cfg = ResNetConditioningConfig(resnet=_small_cfg())
        layer = cfg.build(latent_state_dim=_STATE, latent_time_dim=_TIME)
        x = torch.randn(_BATCH, _STATE)
        t = torch.randn(_BATCH, _TIME)
        assert layer(t, x).shape == (_BATCH, _STATE)
