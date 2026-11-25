from typing import Any

from sckitflow.data import _dims as dims
from sckitflow.data import _group_encoders as group_encoders
from sckitflow.data import _mixins as mixins
from sckitflow.data import _utils as utils
from sckitflow.data import containers, schemas, sim
from sckitflow.data._datamodule import FlowDataModule
from sckitflow.data._dims import DataDimensions
from sckitflow.data._manager import DataManager, DataManagerConfig, LoaderKwargs

__all__ = [
    "FlowDataModule",
    "DataDimensions",
    "containers",
    "dims",
    "group_encoders",
    "mixins",
    "utils",
    "schemas",
    "sim",
    "DataManager",
    "DataManagerConfig",
    "Loader",
    "EvalLoader",
    "LoaderKwargs",
]

_LAZY = frozenset({"Loader", "EvalLoader"})


def __getattr__(name: str) -> Any:
    """Resolve the boundannbatch-backed loaders on first access (PEP 562).

    Importing them costs ~2s of boundannbatch/annbatch/zarr, which nothing that only configures a
    :class:`DataManager` should pay. :class:`LoaderKwargs` stays eager -- it is a plain TypedDict.
    """
    if name in _LAZY:
        from sckitflow.data import _loader

        return getattr(_loader, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
