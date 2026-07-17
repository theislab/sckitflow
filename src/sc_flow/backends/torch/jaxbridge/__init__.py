"""Train CellFlow with JAX for all numerics and torch/Lightning only for optimization.

torch holds the parameters and runs the optimizer; every number (probability path,
velocity field, FM loss, gradient) is CellFlow's JAX code, bridged into torch
autograd via DLPack. See :mod:`._bridge` for the ownership/transfer contract.
"""

from sc_flow.backends.torch.jaxbridge._bridge import (
    JaxLossFunction,
    assert_same_device,
    jax_to_torch,
    torch_to_jax,
)
from sc_flow.backends.torch.jaxbridge._cellflow import CellFlowJaxModule, make_fm_value_and_grad

__all__ = [
    "JaxLossFunction",
    "assert_same_device",
    "jax_to_torch",
    "torch_to_jax",
    "CellFlowJaxModule",
    "make_fm_value_and_grad",
]
