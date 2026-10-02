from typing import Any

from sckitflow.data import _dims as dims
from sckitflow.data import _group_encoders as group_encoders
from sckitflow.data import _mixins as mixins
from sckitflow.data import _utils as utils
from sckitflow.data import containers, schemas, sim, splitters
from sckitflow.data._config import FlowDataModuleConfig
from sckitflow.data._datamodule import FlowDataModule
from sckitflow.data._dims import DataDimensions
from sckitflow.data._manager import DataManager, DataManagerConfig, LoaderKwargs
from sckitflow.data.splitters import CombinationSplitter, CombinationSplitterConfig, Splitter

__all__ = [
    "FlowDataModule",
    "FlowDataModuleConfig",
    "DataDimensions",
    "containers",
    "dims",
    "group_encoders",
    "mixins",
    "utils",
    "schemas",
    "sim",
    "splitters",
    "DataManager",
    "DataManagerConfig",
    "Loader",
    "EvalLoader",
    "LoaderKwargs",
    "Splitter",
    "CombinationSplitter",
    "CombinationSplitterConfig",
]

_LAZY = frozenset({"Loader", "EvalLoader"})


def __getattr__(name: str) -> Any:
    """Resolve the scfit-backed loaders on first access (PEP 562).

    Importing them costs ~2s of scfit/annbatch/zarr, which nothing that only configures a
    :class:`DataManager` should pay. :class:`LoaderKwargs` stays eager -- it is a plain TypedDict.
    """
    if name in _LAZY:
        from sckitflow.data import _loader

        return getattr(_loader, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
