"""Shared Component base classes for the two portable model roles (destined for ``scfit``).

Every model family expresses its architecture and its objective as a :class:`sc_flow.Component`. These two
abstract family bases carry no ``type_id`` (so they stay unregistered and usable as the ``expected`` family
in ``from_spec``); concrete configs — ``GeneEncoderConfig``, ``MLPVelocityConfig``, ``OTFMObjectiveConfig``,
``ContrastiveObjectiveConfig`` — subclass them and register with a ``type_id``.

``build(context)`` turns the portable config into a runtime object: an :class:`torch.nn.Module` for an
architecture, an :class:`sc_flow.training.Objective` for an objective.
"""

from __future__ import annotations

from typing import Any

from sc_flow._registry import Component

__all__ = ["ArchitectureConfig", "ObjectiveConfig"]


class ArchitectureConfig(Component):
    """A portable architecture recipe (unregistered family base). ``build(ctx) -> torch.nn.Module``."""

    def build(self, context: Any = None) -> Any:  # -> torch.nn.Module
        raise NotImplementedError(f"{type(self).__name__} must implement build(self, context).")


class ObjectiveConfig(Component):
    """A portable objective recipe (unregistered family base). ``build(ctx) -> sc_flow.training.Objective``."""

    def build(self, context: Any = None) -> Any:  # -> Objective
        raise NotImplementedError(f"{type(self).__name__} must implement build(self, context).")
