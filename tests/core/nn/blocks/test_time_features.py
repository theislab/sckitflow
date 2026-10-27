"""Tests for the sinusoidal time features and their configs.

Covers:

* ``_ott_jax_time_features`` and ``_torch_cfm_time_features`` directly
  (validation, shape contract, mathematical properties, determinism), and
* ``OTTJaxTimeFeaturesConfig`` / ``TorchCFMTimeFeaturesConfig``
  (``build`` returning a working callable, kwargs forwarding, validation
  propagation, and the ``self.dict()`` / registry-``type`` leak regression
  shared with the pooling and ResNet configs).

Import path: adjust ``sckitflow.core.nn._time_features`` to wherever the
module actually lives.
"""

from __future__ import annotations

import math

import pytest
import torch
from pydantic import ValidationError

from sckitflow.core.nn.blocks._time_features import (
    OTTJaxTimeFeaturesConfig,
    TimeFeaturesConfig,
    TorchCFMTimeFeaturesConfig,
    _ott_jax_time_features,
    _torch_cfm_time_features,
)

# ---------------------------------------------------------------------------
# Shared parametrisation
# ---------------------------------------------------------------------------

EVEN_N = [2, 4, 8]
ODD_OR_NONPOSITIVE_N = [0, -2, 1, 3, 5]
MAX_PERIODS = [1, 10, 100, 10_000]

# Documented input shapes: ``()``, ``(B,)``, ``(B, 1)``.
INPUT_SHAPES = [
    pytest.param((), id="scalar"),
    pytest.param((1,), id="batch-1"),
    pytest.param((5,), id="batch-5"),
    pytest.param((5, 1), id="batch-5-1"),
]


def _make_time_input(shape: tuple[int, ...]) -> torch.Tensor:
    """``torch.rand(*())`` unpacks to zero arguments and raises; use a
    scalar literal for the ``()`` case and vararg unpacking elsewhere."""
    if shape == ():
        return torch.tensor(0.5)
    return torch.rand(*shape)


def _expected_batch(shape: tuple[int, ...]) -> int:
    return 1 if shape in ((), (1,)) else shape[0]


# ---------------------------------------------------------------------------
# _ott_jax_time_features
# ---------------------------------------------------------------------------


class TestOttJaxTimeFeatures:
    # ---- shape contract ---------------------------------------------------

    @pytest.mark.parametrize("n", EVEN_N)
    @pytest.mark.parametrize("shape", INPUT_SHAPES)
    def test_output_shape(self, n: int, shape: tuple[int, ...]) -> None:
        t = _make_time_input(shape)
        out = _ott_jax_time_features(t, n)
        assert out.dim() == 2
        assert out.shape[-1] == n
        assert out.shape[0] == _expected_batch(shape)

    # ---- validation -------------------------------------------------------

    @pytest.mark.parametrize("n", ODD_OR_NONPOSITIVE_N)
    def test_invalid_n_raises(self, n: int) -> None:
        with pytest.raises(ValueError, match="even positive integer"):
            _ott_jax_time_features(torch.tensor(0.5), n)

    # ---- mathematical properties -----------------------------------------

    @pytest.mark.parametrize("n", EVEN_N)
    def test_at_zero_is_cos_column_of_ones_sin_column_of_zeros(self, n: int) -> None:
        out = _ott_jax_time_features(torch.tensor(0.0), n)
        half = n // 2
        assert torch.allclose(out[..., :half], torch.ones_like(out[..., :half]))
        assert torch.allclose(out[..., half:], torch.zeros_like(out[..., half]))

    def test_n_eq_2_always_returns_cos0_sin0(self) -> None:
        """For n=2 there is only frequency 0, so the output is always [1, 0]."""
        for t_val in [0.0, 0.25, 0.5, 1.0, 3.14]:
            out = _ott_jax_time_features(torch.tensor(t_val), 2)
            assert torch.allclose(out, torch.tensor([[1.0, 0.0]]), atol=1e-6)

    def test_n_eq_4_quarter_period(self) -> None:
        """n=4, t=0.25 → freq=[0,2π], t*freq=[0,π/2] → cos=[1,0], sin=[0,1]."""
        out = _ott_jax_time_features(torch.tensor(0.25), 4)
        expected = torch.tensor([[1.0, 0.0, 0.0, 1.0]])
        assert torch.allclose(out, expected, atol=1e-6)

    def test_n_eq_4_half_period(self) -> None:
        """n=4, t=0.5 → freq=[0,2π], t*freq=[0,π] → cos=[1,-1], sin=[0,0]."""
        out = _ott_jax_time_features(torch.tensor(0.5), 4)
        expected = torch.tensor([[1.0, -1.0, 0.0, 0.0]])
        assert torch.allclose(out, expected, atol=1e-6)

    @pytest.mark.parametrize("n", EVEN_N)
    def test_periodicity(self, n: int) -> None:
        """For integer t, 2π·i·t is a multiple of 2π; features are unchanged."""
        out0 = _ott_jax_time_features(torch.tensor(0.0), n)
        for t_int in [1.0, 2.0, 3.0]:
            out_t = _ott_jax_time_features(torch.tensor(t_int), n)
            assert torch.allclose(out0, out_t, atol=1e-6)

    @pytest.mark.parametrize("n", EVEN_N)
    def test_bounded(self, n: int) -> None:
        """cos and sin are always in [-1, 1]."""
        out = _ott_jax_time_features(torch.rand(100), n)
        assert out.abs().max() <= 1.0 + 1e-6

    @pytest.mark.parametrize("n", EVEN_N)
    def test_deterministic(self, n: int) -> None:
        t = torch.tensor([0.3, 0.7, 0.9])
        assert torch.allclose(_ott_jax_time_features(t, n), _ott_jax_time_features(t, n))


# ---------------------------------------------------------------------------
# _torch_cfm_time_features
# ---------------------------------------------------------------------------


class TestTorchCFMTimeFeatures:
    # ---- shape contract ---------------------------------------------------

    @pytest.mark.parametrize("n", EVEN_N)
    @pytest.mark.parametrize("shape", INPUT_SHAPES)
    @pytest.mark.parametrize("max_period", [1, 10_000])
    def test_output_shape(self, n: int, shape: tuple[int, ...], max_period: int) -> None:
        t = _make_time_input(shape)
        out = _torch_cfm_time_features(t, n, max_period)
        assert out.dim() == 2
        assert out.shape[-1] == n
        assert out.shape[0] == _expected_batch(shape)

    # ---- validation -------------------------------------------------------

    @pytest.mark.parametrize("n", ODD_OR_NONPOSITIVE_N)
    def test_invalid_n_raises(self, n: int) -> None:
        with pytest.raises(ValueError, match="even positive integer"):
            _torch_cfm_time_features(torch.tensor(0.5), n, 10_000)

    @pytest.mark.parametrize("max_period", [0, -1])
    def test_invalid_max_period_raises(self, max_period: int) -> None:
        """Non-positive ``max_period`` reaches ``math.log`` and raises."""
        with pytest.raises(ValueError):
            _torch_cfm_time_features(torch.tensor(0.5), 4, max_period)

    # ---- mathematical properties -----------------------------------------

    @pytest.mark.parametrize("max_period", MAX_PERIODS)
    def test_at_zero(self, max_period: int) -> None:
        """At t=0, cos(0)=1 and sin(0)=0 for every frequency."""
        out = _torch_cfm_time_features(torch.tensor(0.0), 4, max_period)
        expected = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        assert torch.allclose(out, expected, atol=1e-6)

    def test_n_eq_2_is_plain_cos_sin(self) -> None:
        """n=2 → a single frequency 0 → the output is exactly [cos(t), sin(t)]."""
        for t_val in [0.0, 0.3, 0.5, 1.2]:
            out = _torch_cfm_time_features(torch.tensor(t_val), 2, 10_000)
            expected = torch.tensor([[math.cos(t_val), math.sin(t_val)]])
            assert torch.allclose(out, expected, atol=1e-6)

    def test_max_period_affects_output(self) -> None:
        t = torch.tensor(0.5)
        assert not torch.allclose(
            _torch_cfm_time_features(t, 4, 10),
            _torch_cfm_time_features(t, 4, 10_000),
        )

    def test_max_period_eq_1_gives_constant_frequency(self) -> None:
        """max_period=1 → log=0 → every frequency is 1."""
        out = _torch_cfm_time_features(torch.tensor(0.5), 4, 1)
        c, s = math.cos(0.5), math.sin(0.5)
        expected = torch.tensor([[c, c, s, s]])
        assert torch.allclose(out, expected, atol=1e-6)

    @pytest.mark.parametrize("n", EVEN_N)
    @pytest.mark.parametrize("max_period", MAX_PERIODS)
    def test_bounded(self, n: int, max_period: int) -> None:
        out = _torch_cfm_time_features(torch.rand(100), n, max_period)
        assert out.abs().max() <= 1.0 + 1e-6

    def test_deterministic(self) -> None:
        t = torch.tensor([0.3, 0.7, 0.9])
        assert torch.allclose(
            _torch_cfm_time_features(t, 8, 10_000),
            _torch_cfm_time_features(t, 8, 10_000),
        )


# ---------------------------------------------------------------------------
# TimeFeaturesConfig (abstract base)
# ---------------------------------------------------------------------------


class TestTimeFeaturesConfigBase:
    def test_base_is_abstract(self) -> None:
        """scfit's ``Component`` treats ``TimeFeaturesConfig`` as a *family
        base* and refuses direct instantiation — the user must pick a
        subclass with a registered ``type``. This documents the constraint
        rather than attempting to instantiate it.
        """
        with pytest.raises(ValidationError):
            TimeFeaturesConfig(n_time_features=4)

    def test_base_is_a_component_subclass(self) -> None:
        from scfit.registry import Component

        assert issubclass(TimeFeaturesConfig, Component)


# ---------------------------------------------------------------------------
# OTTJaxTimeFeaturesConfig
# ---------------------------------------------------------------------------


class TestOTTJaxTimeFeaturesConfig:
    def test_field(self) -> None:
        cfg = OTTJaxTimeFeaturesConfig(n_time_features=4)
        assert cfg.n_time_features == 4

    def test_build_returns_callable(self) -> None:
        cfg = OTTJaxTimeFeaturesConfig(n_time_features=4)
        assert callable(cfg.build())

    def test_build_callable_matches_raw_function(self) -> None:
        cfg = OTTJaxTimeFeaturesConfig(n_time_features=4)
        fn = cfg.build()
        t = torch.tensor([0.25, 0.5, 0.75])
        assert torch.allclose(fn(t), _ott_jax_time_features(t, 4))

    def test_build_callable_shape(self) -> None:
        cfg = OTTJaxTimeFeaturesConfig(n_time_features=8)
        out = cfg.build()(torch.rand(5))
        assert out.shape == (5, 8)

    def test_build_validation_is_deferred_to_call(self) -> None:
        """Odd ``n_time_features`` is accepted at construction; the raw
        function's ``ValueError`` fires when the callable is invoked.
        """
        cfg = OTTJaxTimeFeaturesConfig(n_time_features=3)  # no construction-time check
        fn = cfg.build()
        with pytest.raises(ValueError, match="even positive integer"):
            fn(torch.tensor(0.5))

    def test_build_no_type_leak(self) -> None:
        """Regression guard for the same bug seen in ``_pooling.py`` and
        ``_resnet.py``: ``build()`` calls ``self.dict()``, which includes the
        registry ``type`` field on decorated subclasses, and the raw function
        rejects it at call time.
        """
        cfg = OTTJaxTimeFeaturesConfig(n_time_features=4)
        try:
            fn = cfg.build()
        except TypeError as exc:
            pytest.fail(
                f"build() raised TypeError: {exc}. Likely cause: self.dict() "
                f"includes the registry 'type' field. Fix: use dict(self) or "
                f"model_dump(exclude={{'type'}})."
            )
        try:
            out = fn(torch.tensor(0.5))
        except TypeError as exc:
            pytest.fail(
                f"calling the built function raised TypeError: {exc}. "
                f"Likely cause: self.dict() includes the registry 'type' "
                f"field and the partial forwards it to _ott_jax_time_features."
            )
        assert out.shape == (1, 4)


# ---------------------------------------------------------------------------
# TorchCFMTimeFeaturesConfig
# ---------------------------------------------------------------------------


class TestTorchCFMTimeFeaturesConfig:
    def test_fields(self) -> None:
        cfg = TorchCFMTimeFeaturesConfig(n_time_features=4, max_period=10_000)
        assert cfg.n_time_features == 4
        assert cfg.max_period == 10_000

    def test_build_returns_callable(self) -> None:
        cfg = TorchCFMTimeFeaturesConfig(n_time_features=4, max_period=10_000)
        assert callable(cfg.build())

    def test_build_callable_matches_raw_function(self) -> None:
        cfg = TorchCFMTimeFeaturesConfig(n_time_features=4, max_period=100)
        fn = cfg.build()
        t = torch.tensor([0.25, 0.5, 0.75])
        assert torch.allclose(fn(t), _torch_cfm_time_features(t, 4, 100))

    def test_max_period_is_forwarded(self) -> None:
        cfg_a = TorchCFMTimeFeaturesConfig(n_time_features=4, max_period=10)
        cfg_b = TorchCFMTimeFeaturesConfig(n_time_features=4, max_period=10_000)
        t = torch.tensor(0.5)
        assert not torch.allclose(cfg_a.build()(t), cfg_b.build()(t))

    def test_build_callable_shape(self) -> None:
        cfg = TorchCFMTimeFeaturesConfig(n_time_features=8, max_period=10_000)
        out = cfg.build()(torch.rand(5))
        assert out.shape == (5, 8)

    def test_build_no_type_leak(self) -> None:
        cfg = TorchCFMTimeFeaturesConfig(n_time_features=4, max_period=10_000)
        try:
            fn = cfg.build()
        except TypeError as exc:
            pytest.fail(
                f"build() raised TypeError: {exc}. Likely cause: self.dict() "
                f"includes the registry 'type' field. Fix: use dict(self) or "
                f"model_dump(exclude={{'type'}})."
            )
        try:
            out = fn(torch.tensor(0.5))
        except TypeError as exc:
            pytest.fail(
                f"calling the built function raised TypeError: {exc}. "
                f"Likely cause: self.dict() includes the registry 'type' "
                f"field and the partial forwards it to _torch_cfm_time_features."
            )
        assert out.shape == (1, 4)


# ---------------------------------------------------------------------------
# Cross-cutting
# ---------------------------------------------------------------------------


class TestSharedInvariants:
    def test_fn_is_class_var(self) -> None:
        """``fn`` must not be a pydantic field (it's dispatch, not state)."""
        for cfg in (
            OTTJaxTimeFeaturesConfig(n_time_features=4),
            TorchCFMTimeFeaturesConfig(n_time_features=4, max_period=10_000),
        ):
            assert "fn" not in dict(cfg)
            assert "fn" not in cfg.__class__.model_fields

    def test_ott_and_cfm_produce_different_outputs(self) -> None:
        t = torch.tensor(0.5)
        ott = OTTJaxTimeFeaturesConfig(n_time_features=4).build()(t)
        cfm = TorchCFMTimeFeaturesConfig(n_time_features=4, max_period=10_000).build()(t)
        assert not torch.allclose(ott, cfm)

    def test_expected_public_names(self) -> None:
        from sckitflow.core.nn.blocks import _time_features

        assert set(_time_features.__all__) == {
            "TimeFeaturesFn",
            "TimeFeaturesConfig",
            "OTTJaxTimeFeaturesConfig",
            "TorchCFMTimeFeaturesConfig",
        }
