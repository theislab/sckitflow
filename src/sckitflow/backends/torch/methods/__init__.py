from typing import Literal

from sckitflow.backends.torch.methods._base import TorchBaseMethod, TorchGenerativeFlow
from sckitflow.backends.torch.methods.library._cfm import CFM
from sckitflow.backends.torch.methods.library._cfm_joint import CFM_Joint

METHODS_REGISTRY = {
    "cfm": CFM,
    "cfm-j": CFM_Joint
}
AVAILABLE_METHODS = Literal["cfm", "cfm-j"]

__all__ = ["TorchBaseMethod", "TorchGenerativeFlow", "CFM", "CFM_Joint", "METHODS_REGISTRY", "AVAILABLE_METHODS"]
