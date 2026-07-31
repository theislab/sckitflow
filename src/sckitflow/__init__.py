from importlib.metadata import version

from sckitflow import backends, data, dataset, methods, trainer
from sckitflow._model import Model, ModelBuilder

__version__ = version("sckitflow")

__all__ = [
    "Model",
    "ModelBuilder",
    "__version__",
    "backends",
    "data",
    "dataset",
    "methods",
    "trainer",
]
