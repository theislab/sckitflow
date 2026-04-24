from typing import Literal

from sc_flow.backends.torch.methods._base import TorchBaseMethod, TorchGenerativeFlow
from sc_flow.backends.torch.methods.library._cfm import CFM
from sc_flow.backends.torch.methods.library._emd import EMD
from sc_flow.backends.torch.methods.library._fmm import FMM
from sc_flow.backends.torch.methods.library._lmd import LMD

METHODS_REGISTRY = {
    "cfm": CFM,
    "emd": EMD,
    "fmm": FMM,
    "lmd": LMD,
}
AVAILABLE_METHODS = Literal["cfm"]

__all__ = ["TorchBaseMethod", "TorchGenerativeFlow", "CFM", "METHODS_REGISTRY", "AVAILABLE_METHODS"]
