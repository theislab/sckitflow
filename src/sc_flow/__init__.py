"""sc_flow top-level package.

The active path is torch + PyTorch-Lightning: ``FlowMatching`` (the model) over the ``data`` streaming
layer, with the torch numerics under ``backends``. ``data`` is imported eagerly; ``backends`` and
``FlowMatching`` are exposed lazily so ``import sc_flow.data`` works without pulling torch/jax.
Subsystems not on the train path are quarantined under ``sc_flow.legacy``.
"""

from sc_flow import data
from sc_flow._optional import require

__all__ = [
    "FlowMatching",
    "backends",
    "data",
]

_LAZY_SUBMODULES = frozenset({"backends"})


def __getattr__(name: str):
    # The model + backend subsystems import the optional heavy deps (torch/jax/lightning). Route their
    # lazy import through require() so a bare env gets a clear "install sc-flow-tools[...]" hint instead
    # of a raw ModuleNotFoundError deep in a traceback.
    if name in _LAZY_SUBMODULES:
        return require(f"sc_flow.{name}")
    if name == "FlowMatching":
        return require("sc_flow._model").FlowMatching
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
