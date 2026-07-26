"""Back-compat shim: moved to :mod:`scfit.training._objective`."""

from scfit.training._objective import (
    OBJECTIVE_REGISTRY,
    Objective,
    build_objective,
    register_objective,
)

__all__ = ["Objective", "register_objective", "build_objective", "OBJECTIVE_REGISTRY"]
