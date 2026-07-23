
from sc_flow._optional import require
from sc_flow import data

__all__ = [
    "FlowMatching",
    "FlowMatchingConfig",
    "data",
    "flow",
]

_LAZY_SUBMODULES = frozenset({"flow"})


def __getattr__(name: str):
    # The model + flow subsystems import the optional heavy deps (torch/jax/lightning). Route their lazy
    # import through require() so a bare env gets a clear "install sc-flow-tools[...]" hint instead of a
    # raw ModuleNotFoundError deep in a traceback.
    if name in _LAZY_SUBMODULES:
        return require(f"sc_flow.{name}")
    if name in ("FlowMatching", "FlowMatchingConfig"):
        return getattr(require("sc_flow._model"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
