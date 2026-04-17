from typing import Literal

from sc_flow.backends.torch.methods._base import BaseGenerativeFlow
from sc_flow.backends.torch.methods._methods import CFM

METHODS_REGISTRY = {
    "cfm": CFM,
}
AVAILABLE_METHODS = Literal["cfm"]

__all__ = ["BaseGenerativeFlow", "CFM", "METHODS_REGISTRY", "AVAILABLE_METHODS"]
