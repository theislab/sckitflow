from typing import Literal

from sckitflow.core.methods._base import BaseMethod, GenerativeFlow
from sckitflow.core.methods._custom import register_method
from sckitflow.core.methods._opt import OptimConfig, OptimizationManager
from sckitflow.core.methods.library._cfm import CFM

METHODS_REGISTRY = {
    "cfm": CFM,
}
AVAILABLE_METHODS = Literal["cfm"]

__all__ = [
    "BaseMethod",
    "GenerativeFlow",
    "CFM",
    "OptimConfig",
    "OptimizationManager",
    "register_method",
    "METHODS_REGISTRY",
    "AVAILABLE_METHODS",
]
