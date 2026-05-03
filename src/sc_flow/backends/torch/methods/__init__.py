from typing import Literal

from sc_flow.backends.torch.methods._base import TorchBaseMethod, TorchGenerativeFlow
from sc_flow.backends.torch.methods.library._cfm import CFM
from sc_flow.backends.torch.methods.library._csm import CSM

METHODS_REGISTRY = {
    "cfm": CFM,
    "csm": CSM,
}
AVAILABLE_METHODS = Literal["cfm"]

__all__ = ["TorchBaseMethod", "TorchGenerativeFlow", "CFM", "METHODS_REGISTRY", "AVAILABLE_METHODS"]
