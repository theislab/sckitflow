"""Tests for :class:`ModuleConfig`.

``ModuleConfig`` is the family base for configs that build a
:class:`torch.nn.Module` from :class:`~sckitflow.data._dims.DataDimensions`.
At present :class:`SetEncoderConfig` is the only concrete subclass; it's
used here as a probe because ``ModuleConfig`` itself is a family base and
cannot be instantiated directly.

The contract tested here:

* ``ModuleConfig`` is a ``Component`` subclass (so ``@component`` can
  recognise descendants, and ``@component``-decorated subclasses remain
  instantiable).
* Direct instantiation is rejected (family base).
* ``build(dims)`` is declared on the base; subclasses override it.
* ``build`` returns a ``torch.nn.Module``.
* Subclass return types can narrow covariantly — ``SetEncoderConfig.build``
  returns ``SetEncoder``, which is a ``Module`` subtype.
* Metaclass composition (``Protocol`` + ``Component`` + pydantic) does not
  break subclass construction.

If ``ModuleConfig`` is ever refactored (e.g. from ``Protocol + Component``
to ``Component + ABC``), the ``TestModuleConfigBaseClassFlavor`` class
adapts to the change automatically; the rest of the file only pins the
behavioural contract.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sckitflow.core.nn._config import ModuleConfig
from sckitflow.core.nn.blocks._mlp import MLPConfig, MLPEncoderConfig
from sckitflow.core.nn.composite._set_encoder import SetEncoder, SetEncoderConfig

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _dims(*, categorical=("condition0",)) -> SimpleNamespace:
    """Minimal ``DataDimensions`` substitute for ``SetEncoderConfig.build``.

    Only ``condition_continuous_dims`` and ``condition_reps_dims`` are read
    by the set encoder constructor.
    """
    return SimpleNamespace(
        condition_continuous_dims=None,
        condition_reps_dims=dict.fromkeys(categorical, 4),
    )


def _set_encoder_cfg() -> SetEncoderConfig:
    return SetEncoderConfig(
        output_dim=16,
        categorical_input_layers={
            "condition0": MLPEncoderConfig(mlp=MLPConfig(), output_dim=2),
        },
    )


# ---------------------------------------------------------------------------
# Base-class contract
# ---------------------------------------------------------------------------


class TestModuleConfigBase:
    def test_is_a_component(self) -> None:
        """``ModuleConfig`` must be a ``Component`` subclass, otherwise the
        ``@component`` decorator won't recognise descendants and pydantic
        field declaration (``output_dim: int``, ...) won't apply.
        """
        from scfit.registry import Component

        assert issubclass(ModuleConfig, Component)

    def test_cannot_instantiate_directly(self) -> None:
        """Family-base check: scfit's ``Component`` rejects direct
        instantiation of a non-``@component``-decorated base. This is the
        mechanism that makes ``ModuleConfig`` non-instantiable, not
        ``Protocol``.
        """
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ModuleConfig()  # type: ignore[abstract]

    def test_build_method_is_declared(self) -> None:
        """``build`` must exist on the class so subclasses can override it."""
        assert hasattr(ModuleConfig, "build")
        assert callable(ModuleConfig.build)

    def test_build_signature_takes_dims(self) -> None:
        """The second parameter is ``dims`` (after ``self``). Verified via
        inspection so the contract is visible if the method is re-typed.
        """
        import inspect

        sig = inspect.signature(ModuleConfig.build)
        params = list(sig.parameters)
        assert len(params) == 2
        assert params[0] == "self"
        assert params[1] == "dims"

    def test_build_return_annotation_is_nn_module(self) -> None:
        """The base declares ``-> torch.nn.Module``; subclasses may narrow
        covariantly. Pinning the annotation makes an accidental widening
        (e.g. to ``Any``) visible.
        """
        import inspect

        sig = inspect.signature(ModuleConfig.build)
        # ``from __future__ import annotations`` is used in the source, so
        # the annotation may be the string ``"torch.nn.Module"``.
        ann = sig.return_annotation
        assert ann is torch.nn.Module or ann == "torch.nn.Module"


# ---------------------------------------------------------------------------
# Subclass contract — via SetEncoderConfig as probe
# ---------------------------------------------------------------------------


class TestModuleConfigViaSetEncoder:
    """``SetEncoderConfig`` is the only concrete subclass today. It's used
    here as a probe to validate the base class contract end-to-end.
    """

    def test_subclass_inherits_from_base(self) -> None:
        assert issubclass(SetEncoderConfig, ModuleConfig)

    def test_subclass_instantiates(self) -> None:
        """Metaclass composition must not break subclass construction. If
        ``Protocol`` + ``Component`` + pydantic broke under
        ``@component``, this is where it would show.
        """
        cfg = _set_encoder_cfg()
        assert isinstance(cfg, SetEncoderConfig)

    def test_build_is_overridden(self) -> None:
        """The subclass's ``build`` must be its own, not inherited from the
        base. This catches a botched MRO.
        """
        assert SetEncoderConfig.build is not ModuleConfig.build

    def test_build_returns_torch_module(self) -> None:
        """End-to-end contract: ``build(dims)`` returns an ``nn.Module``
        from the base's declared return type.
        """
        cfg = _set_encoder_cfg()
        module = cfg.build(_dims())
        assert isinstance(module, torch.nn.Module)

    def test_build_return_is_covariantly_narrowed(self) -> None:
        """``SetEncoderConfig.build`` returns ``SetEncoder`` — a legitimate
        covariant narrowing of the base's ``torch.nn.Module`` return type.
        """
        cfg = _set_encoder_cfg()
        module = cfg.build(_dims())
        assert isinstance(module, SetEncoder)
        assert isinstance(module, torch.nn.Module)

    def test_built_module_runs_forward(self) -> None:
        """``build(dims)`` must produce a module that actually runs —
        contract tests are worthless if the module is malformed.
        """
        cfg = _set_encoder_cfg()
        module = cfg.build(_dims()).eval()
        cond = {"condition0": torch.randn(4, 3, 4)}
        with torch.no_grad():
            out = module(cond)
        assert out.shape == (4, cfg.output_dim)


# ---------------------------------------------------------------------------
# Base-class flavor probe (Protocol vs ABC vs plain class)
# ---------------------------------------------------------------------------


class TestModuleConfigBaseClassFlavor:
    """Documents the current implementation flavour and adjusts the
    ``isinstance`` expectation to match.

    * ``Protocol`` without ``@runtime_checkable`` → ``isinstance`` raises.
    * ``ABC`` or plain class                    → ``isinstance`` works.

    This lets the tests survive a refactor from ``Protocol + Component``
    to ``Component + ABC`` without manual edits.
    """

    def test_isinstance_behaviour_matches_flavor(self) -> None:
        from typing import Protocol

        cfg = _set_encoder_cfg()
        if issubclass(ModuleConfig, Protocol):
            with pytest.raises(TypeError, match="runtime_checkable"):
                isinstance(cfg, ModuleConfig)
        else:
            assert isinstance(cfg, ModuleConfig)

    def test_not_decorated_runtime_checkable(self) -> None:
        """``@runtime_checkable`` is not applied; adding it would silently
        change ``isinstance`` semantics. Pin the current state so the
        change is a visible test failure rather than a subtle behaviour
        shift.
        """
        from typing import Protocol

        if not issubclass(ModuleConfig, Protocol):
            pytest.skip("ModuleConfig is no longer a Protocol")
        assert not getattr(ModuleConfig, "_is_runtime_protocol", False)


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------


class TestModuleConfigRegistry:
    def test_subclass_is_registered(self) -> None:
        """``SetEncoderConfig`` is ``@component``-decorated; confirm the
        registration survives inheriting from ``ModuleConfig`` (this is
        what breaks if ``ModuleConfig``'s metaclass interferes with the
        decorator).
        """

        # The exact registry API varies; check via the metadata that the
        # decorator sets. ``Component.builds`` is the one all other configs
        # in the tree rely on.
        assert getattr(SetEncoderConfig, "builds", None) is SetEncoder

    def test_subclass_round_trips_through_pydantic(self) -> None:
        """Pydantic's ``model_dump`` / ``model_validate`` must work through
        the ``ModuleConfig`` base. A malformed MRO would surface here.
        """
        cfg = _set_encoder_cfg()
        payload = cfg.model_dump()
        rebuilt = SetEncoderConfig.model_validate(payload)
        assert rebuilt.output_dim == cfg.output_dim
        assert set(rebuilt.categorical_input_layers) == {"condition0"}
