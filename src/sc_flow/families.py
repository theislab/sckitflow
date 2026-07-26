"""The model-**family** registry + discovery — the ``model.family`` dispatch (destined for ``scfit``).

A *family* is a training paradigm (flow matching, contrastive foundation). Concretely it is
just a callable ``(recipe) -> FamilyBuilder`` — and a **builder class is exactly that** (constructing it
*is* building), so a family needs no wrapper class: the entry point points straight at the builder.

Families register two ways, both without the app importing the plugin:
- **entry points** — a plugin declares ``[project.entry-points."scfit.families"] <name> = "<module>:<Builder>"``;
  :func:`available_families` reads the NAMES **torch-free** (no ``ep.load()``), :func:`load_family` loads just
  the one requested.
- **manual** — :func:`register_family` (name + factory) for editable dev trees / same-process registration.

This module imports only the stdlib, so enumerating families never drags in torch/jax.
"""

from __future__ import annotations

import importlib.metadata
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "FamilyBuilder",
    "FamilyFactory",
    "FAMILY_REGISTRY",
    "register_family",
    "available_families",
    "load_family",
    "build_family",
]

ENTRY_POINT_GROUP = "scfit.families"


@runtime_checkable
class FamilyBuilder(Protocol):
    """What a family's ``build(recipe)`` returns — the pieces an app hands to ``lightning.Trainer.fit``.

    Required: ``module`` / ``datamodule`` / ``callbacks`` / ``save``. Optional, read by the app with a
    default so a pre-contract builder still works:

    * ``metrics_history: dict`` — validation history (feeds the sweep objective + the run report),
    * ``trainer_overrides: dict`` — extra ``Trainer`` kwargs the family's own eval needs (e.g.
      ``val_check_interval`` / ``limit_val_batches``).

    Every first-party family (flow_matching, foundation) implements all of them, so the app drives each
    identically — no family is special-cased at the application level.
    """

    module: Any  # a lightning.LightningModule (e.g. sc_flow.training.TrainingModule)
    datamodule: Any  # a lightning.LightningDataModule (or None + explicit dataloaders)
    callbacks: list

    def save(self, path: Any) -> None: ...


#: A family factory: any callable that turns a recipe into a :class:`FamilyBuilder`. A builder *class*
#: (``FlowMatching`` / ``FoundationModel`` / ``PanCellFlow``) is one — its constructor does the build.
FamilyFactory = Callable[[dict[str, Any]], FamilyBuilder]

FAMILY_REGISTRY: dict[str, FamilyFactory] = {}


def register_family(name: str, family: FamilyFactory) -> FamilyFactory:
    """Register a family factory under ``name`` (same-process / editable trees; entry points need no call).

    Idempotent for the same object; a *different* factory for a known name errors."""
    existing = FAMILY_REGISTRY.get(name)
    if existing is not None and existing is not family:
        raise ValueError(f"family {name!r} already registered to {existing!r}.")
    FAMILY_REGISTRY[name] = family
    return family


def _entry_point_names() -> set[str]:
    try:
        return {ep.name for ep in importlib.metadata.entry_points(group=ENTRY_POINT_GROUP)}
    except Exception:  # noqa: BLE001 — no entry points / odd metadata: fall back to manual registrations
        return set()


def available_families() -> list[str]:
    """Every family name discoverable now — entry points (not loaded) + manual registrations. Torch-free."""
    return sorted(_entry_point_names() | set(FAMILY_REGISTRY))


def load_family(name: str) -> FamilyFactory:
    """Return the family factory, loading its plugin (``ep.load()``) on first use. Actionable error if absent."""
    if name in FAMILY_REGISTRY:
        return FAMILY_REGISTRY[name]
    for ep in importlib.metadata.entry_points(group=ENTRY_POINT_GROUP):
        if ep.name == name:
            return register_family(name, ep.load())
    raise KeyError(f"family {name!r} not found; available: {available_families()}.")


def build_family(name: str, recipe: dict[str, Any]) -> FamilyBuilder:
    """Discover + build a family in one call: ``load_family(name)(recipe)``."""
    return load_family(name)(recipe)
