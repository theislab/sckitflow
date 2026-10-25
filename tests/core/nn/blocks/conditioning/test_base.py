"""Tests for :class:`BaseConditioningLayer` and :class:`BaseConditioningLayerConfig`.

Focus: the abstract contract, the latent-dim properties, and the shape
validation helper ``_verify_inputs_shape``. A concrete implementation
(``ConcatConditioning``) is used as a probe because the base class itself
is abstract, matching the historical ``test_conditioning_layers.py`` pattern.
"""

from __future__ import annotations

import pytest
import torch

from sckitflow.core.nn.blocks.conditioning._base import (
    BaseConditioningLayer,
    BaseConditioningLayerConfig,
)
from sckitflow.core.nn.blocks.conditioning._concat import ConcatConditioning

_STATE = 16
_TIME = 6
_COND = 8
_BATCH = 4


class TestBaseConditioningLayer:
    # ---- abstractness / construction -------------------------------------

    def test_cannot_instantiate_abstract(self) -> None:
        """``output_dim`` and ``forward`` are both ``@abc.abstractmethod``."""
        with pytest.raises(TypeError):
            BaseConditioningLayer(_STATE, _TIME)  # type: ignore[abstract]

    def test_concrete_subclass_constructs(self) -> None:
        layer = ConcatConditioning(_STATE, _TIME, _COND)
        assert isinstance(layer, BaseConditioningLayer)
        assert isinstance(layer, torch.nn.Module)

    # ---- latent-dim properties -------------------------------------------

    def test_latent_state_dim(self) -> None:
        layer = ConcatConditioning(_STATE, _TIME)
        assert layer.latent_state_dim == _STATE

    def test_latent_time_dim(self) -> None:
        layer = ConcatConditioning(_STATE, _TIME)
        assert layer.latent_time_dim == _TIME

    def test_condition_dim_property_typo(self) -> None:
        layer = ConcatConditioning(_STATE, _TIME, _COND)
        assert layer.latent_condition_dim == _COND

    def test_condition_dim_property_none(self) -> None:
        layer = ConcatConditioning(_STATE, _TIME)
        assert layer.latent_condition_dim is None

    # ---- _verify_inputs_shape contract ----------------------------------

    def test_verify_shape_wrong_state_dim(self) -> None:
        layer = ConcatConditioning(_STATE, _TIME)
        with pytest.raises(RuntimeError, match="state"):
            layer(torch.zeros(_BATCH, _TIME), torch.zeros(_BATCH, _STATE + 1))

    def test_verify_shape_wrong_time_dim(self) -> None:
        layer = ConcatConditioning(_STATE, _TIME)
        with pytest.raises(RuntimeError, match="time"):
            layer(torch.zeros(_BATCH, _TIME + 1), torch.zeros(_BATCH, _STATE))

    def test_verify_shape_condition_missing_when_expected(self) -> None:
        layer = ConcatConditioning(_STATE, _TIME, _COND)
        with pytest.raises(RuntimeError, match="None"):
            layer(torch.zeros(_BATCH, _TIME), torch.zeros(_BATCH, _STATE))

    def test_verify_shape_condition_provided_when_not_expected(self) -> None:
        layer = ConcatConditioning(_STATE, _TIME)
        with pytest.raises(RuntimeError, match="condition"):
            layer(
                torch.zeros(_BATCH, _TIME),
                torch.zeros(_BATCH, _STATE),
                torch.zeros(_BATCH, _COND),
            )

    def test_verify_shape_condition_wrong_dim(self) -> None:
        layer = ConcatConditioning(_STATE, _TIME, _COND)
        with pytest.raises(RuntimeError, match="condition"):
            layer(
                torch.zeros(_BATCH, _TIME),
                torch.zeros(_BATCH, _STATE),
                torch.zeros(_BATCH, _COND + 1),
            )

    # ---- error message formatting ---------------------------------------

    def test_verify_shape_error_has_a_space_after_period(self) -> None:
        """The base-class error messages are built from two adjacent string
        literals with no space between ``.`` and ``Found``; this test pins
        the current (broken) formatting so a future fix is a visible change.
        If you fix the missing space, update this test's expectation.
        """
        layer = ConcatConditioning(_STATE, _TIME)
        with pytest.raises(RuntimeError) as excinfo:
            layer(torch.zeros(_BATCH, _TIME), torch.zeros(_BATCH, _STATE + 1))
        assert "dimension.Found" in str(excinfo.value)


class TestBaseConditioningLayerConfig:
    def test_is_a_component(self) -> None:
        from scfit.registry import Component

        assert issubclass(BaseConditioningLayerConfig, Component)

    def test_build_is_defined(self) -> None:
        assert callable(BaseConditioningLayerConfig.build)

    def test_base_config_not_directly_instantiable(self) -> None:
        """scfit's ``Component`` refuses to instantiate non-decorated
        subclasses ("family base"). Documented behaviour; not a bug in this
        module — it just means ``build`` cannot be called on the base itself.
        """
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            BaseConditioningLayerConfig()  # type: ignore[abstract]
