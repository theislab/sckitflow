"""sc_flow top-level package.

Only the ``data`` layer is imported eagerly. After the ``binded`` migration the
``backends`` / ``dataset`` / ``methods`` / ``trainer`` subsystems (and ``SCFlow``) still
reference symbols removed during the data-layer strip and are being rewired; they are exposed
lazily so that ``import sc_flow.data`` works in isolation while accessing a not-yet-rewired
subsystem raises its real import error at access time.
"""

import importlib

from sc_flow import data

__all__ = [
    "backends",
    "data",
    "dataset",
    "methods",
    "SCFlow",
    "trainer",
]

_LAZY_SUBMODULES = frozenset({"backends", "dataset", "methods", "trainer"})


def __getattr__(name: str):
    if name in _LAZY_SUBMODULES:
        return importlib.import_module(f"sc_flow.{name}")
    if name == "SCFlow":
        from sc_flow._model import SCFlow

        return SCFlow
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
