"""The inference seam: a trained model + a predictor (maps a batch to a prediction).

Symmetric to :class:`~sc_flow.core.training._objective.Objective` (the train-time seam). Where an
``Objective`` turns a batch into a training loss, a :class:`Predictor` turns a trained model + an input
batch into a prediction. The generic :class:`~sc_flow.core.training._harness.TrainingModule` uses the same
predictor for its held-out validation loop, so a validation metric reflects exactly what inference does;
extensions register their own (e.g. the flow-matching ODE integrator).
"""

from __future__ import annotations

import abc
from collections.abc import Callable
from typing import Any

import torch

__all__ = [
    "Predictor",
    "register_predictor",
    "build_predictor",
    "PREDICTOR_REGISTRY",
]


class Predictor(abc.ABC):
    """Maps a trained model + an input batch to a prediction (the inference seam).

    Pure inference: read the model's inputs from ``batch`` (e.g. ``"source"`` / ``"condition"``) and return
    the predicted output. The caller (harness or a public ``predict``) owns everything around it — the
    ground-truth ``target``, metrics, device placement, train/eval mode.
    """

    @abc.abstractmethod
    def predict(self, model: torch.nn.Module, batch: Any) -> torch.Tensor:
        """Return the model's prediction for ``batch`` as a torch tensor."""


PREDICTOR_REGISTRY: dict[str, Callable[..., Predictor]] = {}


def register_predictor(name: str) -> Callable[[Callable[..., Predictor]], Callable[..., Predictor]]:
    """Register a :class:`Predictor` builder under ``name`` (e.g. ``"ode"``)."""

    def deco(builder: Callable[..., Predictor]) -> Callable[..., Predictor]:
        if name in PREDICTOR_REGISTRY:
            raise ValueError(f"Predictor {name!r} already registered.")
        PREDICTOR_REGISTRY[name] = builder
        return builder

    return deco


def build_predictor(name: str, *args: Any, **kwargs: Any) -> Predictor:
    """Instantiate a registered predictor by name.

    Concrete predictors live in the model extension (e.g. :mod:`sc_flow.flow`) and register on import;
    import that layer before building by name.
    """
    if name not in PREDICTOR_REGISTRY:
        raise KeyError(f"Predictor {name!r} not registered. Available: {sorted(PREDICTOR_REGISTRY)}.")
    return PREDICTOR_REGISTRY[name](*args, **kwargs)
