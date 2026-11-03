from sckitflow.core.surrogate._config import (
    CrossEntropyPotentialConfig,
    PotentialConfig,
    SquaredErrorPotentialConfig,
    cross_entropy,
    squared_error,
)
from sckitflow.core.surrogate._potential import SurrogatePotential, SurrogateResponse

__all__ = [
    "SurrogateResponse",
    "SurrogatePotential",
    "PotentialConfig",
    "SquaredErrorPotentialConfig",
    "CrossEntropyPotentialConfig",
    "squared_error",
    "cross_entropy",
]
