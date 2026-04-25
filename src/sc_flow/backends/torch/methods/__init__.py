from typing import Literal

from sc_flow.backends.torch.methods._base import TorchBaseMethod, TorchGenerativeFlow
from sc_flow.backends.torch.methods.library._cfm import CFM
from sc_flow.backends.torch.methods.library._fmm import FMM

METHODS_REGISTRY = {
    "cfm": CFM,
    "fmm": FMM,
}
AVAILABLE_METHODS = Literal["cfm"]

__all__ = [
    "TorchBaseMethod",
    "TorchGenerativeFlow",
    "CFM",
    "FMM",
    "METHODS_REGISTRY",
    "AVAILABLE_METHODS",
]
