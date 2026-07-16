"""Structured configuration for building and training :class:`~sc_flow._model.SCFlow`."""

from sc_flow.config._capabilities import MethodCapabilities
from sc_flow.config._resolve import (
    resolve_callbacks,
    resolve_flow_solver_cls,
    resolve_metrics,
    resolve_probability_path,
)
from sc_flow.config._run import DataConfig, MethodConfig, RunConfig, TrainerConfig

__all__ = [
    "MethodCapabilities",
    "DataConfig",
    "MethodConfig",
    "RunConfig",
    "TrainerConfig",
    "resolve_probability_path",
    "resolve_flow_solver_cls",
    "resolve_metrics",
    "resolve_callbacks",
]
