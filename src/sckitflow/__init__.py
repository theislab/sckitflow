from importlib.metadata import version

from sckitflow import core, data, dataset, methods, trainer
from sckitflow._model import Model, ModelBuilder

__version__ = version("sckitflow")

__all__ = [
    "Model",
    "ModelBuilder",
    "__version__",
    "core",
    "data",
    "dataset",
    "methods",
    "trainer",
]
