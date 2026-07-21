"""sc_flow top-level package.

Two sibling layers: :mod:`sc_flow.core` (the ML-toolbox base — data streaming, the Lightning harness,
optimizer, generic nn + metrics; torch only) and :mod:`sc_flow.flow` (the flow-matching toolbox —
velocity fields, probability paths, objectives, predict, and the optional JAX/OTT coupling bridge).
``FlowMatching`` is the facade wiring them. ``core.data`` is imported eagerly; ``core`` / ``flow`` /
``FlowMatching`` are exposed lazily so ``import sc_flow.core.data`` works without pulling torch/jax.
Subsystems not on the train path are quarantined under ``sc_flow.legacy``.
"""

from sc_flow.core import data
from sc_flow._optional import require

__all__ = [
    "FlowMatching",
    "core",
    "data",
    "flow",
]

_LAZY_SUBMODULES = frozenset({"core", "flow"})


def __getattr__(name: str):
    # The model + flow subsystems import the optional heavy deps (torch/jax/lightning). Route their lazy
    # import through require() so a bare env gets a clear "install sc-flow-tools[...]" hint instead of a
    # raw ModuleNotFoundError deep in a traceback.
    if name in _LAZY_SUBMODULES:
        return require(f"sc_flow.{name}")
    if name == "FlowMatching":
        return require("sc_flow._model").FlowMatching
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
