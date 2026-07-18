"""sc_flow top-level package.

Only the ``data`` layer is imported eagerly. After the ``binded`` migration the
``backends`` / ``dataset`` / ``methods`` / ``trainer`` subsystems (and ``SCFlow``) still
reference symbols removed during the data-layer strip and are being rewired; they are exposed
lazily so that ``import sc_flow.data`` works in isolation while accessing a not-yet-rewired
subsystem raises its real import error at access time.
"""

from sc_flow import data
from sc_flow._optional import require

__all__ = [
    "backends",
    "data",
    "dataset",
    "methods",
    "FlowMatching",
    "trainer",
]

_LAZY_SUBMODULES = frozenset({"backends", "dataset", "methods", "trainer"})


def __getattr__(name: str):
    # The model + backend subsystems import the optional heavy deps (torch/jax/lightning). Route their
    # lazy import through require() so a bare env gets a clear "install sc-flow-tools[...]" hint instead
    # of a raw ModuleNotFoundError deep in a traceback.
    if name in _LAZY_SUBMODULES:
        return require(f"sc_flow.{name}")
    if name == "FlowMatching":
        return require("sc_flow._model").FlowMatching
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
