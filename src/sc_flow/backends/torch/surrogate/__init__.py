"""Differentiable surrogates + potentials for inverse problems.

Wrap a trained model as a differentiable ``x -> response`` map
(:class:`GenerativeFlowSurrogateWrapper`) and score its response against a target
with a :class:`SurrogatePotential` (e.g. :class:`SquaredErrorPotential`). Optimizing
the input against the potential, with the model frozen, solves the inverse problem.
"""

from sc_flow.backends.torch.surrogate._base import SquaredErrorPotential, SurrogatePotential
from sc_flow.backends.torch.surrogate._wrappers import (
    BaseSurrogateWrapper,
    GenerativeFlowSurrogateWrapper,
)

__all__ = [
    "SurrogatePotential",
    "SquaredErrorPotential",
    "BaseSurrogateWrapper",
    "GenerativeFlowSurrogateWrapper",
]
