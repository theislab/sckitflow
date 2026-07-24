from __future__ import annotations

import abc
from typing import Any

import torch

__all__ = ["Predictor"]


class Predictor(abc.ABC):
    """The inference seam: turn a trained model + an input batch into a prediction.

    This is the single inference implementation for a model family — the same object serves external
    prediction and held-out validation, so a validation metric reflects exactly what inference does. Each
    model family (flow matching, a foundation model, ...) writes its own concrete ``Predictor``; the
    downstream toolbox instantiates it directly (there is no name registry — a predictor is always chosen
    by the toolbox that owns the model, not selected globally by string).
    """

    @abc.abstractmethod
    def predict(self, model: torch.nn.Module, batch: Any) -> torch.Tensor:
        """Return the prediction tensor. The hot path — validation calls this."""
        ...

    def predict_with_aux(self, model: torch.nn.Module, batch: Any) -> tuple[torch.Tensor, dict[str, Any]]:
        """Return ``(prediction, aux)`` where ``aux`` carries optional extras (trajectories, per-step stats).

        Default: no aux. Predictors that can produce extras override this; the returned prediction must
        equal ``predict(model, batch)``.
        """
        return self.predict(model, batch), {}
