from importlib.metadata import version

from sckitflow import core, data, dataset, trainer
from sckitflow._predict import predict_adata
from sckitflow._run import Run, RunConfig, load_run, save_run
from sckitflow.core import methods, probability_paths

__version__ = version("sckitflow")

__all__ = [
    "predict_adata",
    "save_run",
    "load_run",
    "Run",
    "RunConfig",
    "__version__",
    "core",
    "data",
    "dataset",
    "trainer",
]
