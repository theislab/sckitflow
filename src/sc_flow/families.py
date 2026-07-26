"""The model-**family** registry + discovery — the ``model.family`` dispatch (destined for ``scfit``).

A *family* is a training paradigm (flow matching, contrastive foundation, pan-cell flow). Each is a
:class:`ModelFamily` whose ``build(recipe)`` returns a builder exposing ``.module`` / ``.datamodule`` /
``.callbacks`` / ``.save`` — the contract ``FlowMatching`` and ``FoundationModel`` already satisfy, and the
one an app (cf-train) drives with a plain ``lightning.Trainer.fit``.

Families register two ways, both without the app importing the plugin:
- **entry points** — a plugin declares ``[project.entry-points."scfit.families"]``; :func:`available_families`
  reads them **torch-free** (no ``ep.load()``), :func:`load_family` loads just the one requested.
- **manual** — :func:`register_family` for editable dev trees / same-process registration.

This module imports only the stdlib, so enumerating families never drags in torch/jax.
"""

from __future__ import annotations

import importlib.metadata
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "ModelFamily",
    "FamilyBuilder",
    "FAMILY_REGISTRY",
    "register_family",
    "available_families",
    "load_family",
    "build_family",
]

ENTRY_POINT_GROUP = "scfit.families"


@runtime_checkable
class FamilyBuilder(Protocol):
    """What a family's ``build(recipe)`` returns — the pieces an app hands to ``lightning.Trainer.fit``."""

    module: Any  # a lightning.LightningModule (e.g. sc_flow.training.TrainingModule)
    datamodule: Any  # a lightning.LightningDataModule (or None + explicit dataloaders)
    callbacks: list

    def save(self, path: Any) -> None: ...


class ModelFamily:
    """A training paradigm. Subclass, set ``name``, implement ``build(recipe) -> FamilyBuilder``."""

    name: str

    def build(self, recipe: dict[str, Any]) -> FamilyBuilder:
        raise NotImplementedError


FAMILY_REGISTRY: dict[str, ModelFamily] = {}


def register_family(family: ModelFamily) -> ModelFamily:
    """Register a family instance (idempotent for the same object; a different object for a known name errors)."""
    existing = FAMILY_REGISTRY.get(family.name)
    if existing is not None and existing is not family:
        raise ValueError(f"family {family.name!r} already registered to {type(existing).__name__}.")
    FAMILY_REGISTRY[family.name] = family
    return family


def _entry_point_names() -> set[str]:
    try:
        return {ep.name for ep in importlib.metadata.entry_points(group=ENTRY_POINT_GROUP)}
    except Exception:  # noqa: BLE001 — no entry points / odd metadata: fall back to manual registrations
        return set()


def available_families() -> list[str]:
    """Every family name discoverable now — entry points (not loaded) + manual registrations. Torch-free."""
    return sorted(_entry_point_names() | set(FAMILY_REGISTRY))


def load_family(name: str) -> ModelFamily:
    """Return the family, loading its plugin (``ep.load()``) on first use. Actionable error if absent."""
    if name in FAMILY_REGISTRY:
        return FAMILY_REGISTRY[name]
    for ep in importlib.metadata.entry_points(group=ENTRY_POINT_GROUP):
        if ep.name == name:
            return register_family(ep.load())
    raise KeyError(f"family {name!r} not found; available: {available_families()}.")


def build_family(name: str, recipe: dict[str, Any]) -> FamilyBuilder:
    """Discover + build a family in one call: ``load_family(name).build(recipe)``."""
    return load_family(name).build(recipe)
