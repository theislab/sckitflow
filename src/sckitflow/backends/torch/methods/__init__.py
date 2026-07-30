from typing import Literal

from sckitflow.backends.torch.methods._base import TorchBaseMethod, TorchGenerativeFlow
from sckitflow.backends.torch.methods.library._cfm import CFM

METHODS_REGISTRY = {
    "cfm": CFM,
}
AVAILABLE_METHODS = Literal["cfm"]

__all__ = ["TorchBaseMethod", "TorchGenerativeFlow", "CFM", "METHODS_REGISTRY", "AVAILABLE_METHODS"]
