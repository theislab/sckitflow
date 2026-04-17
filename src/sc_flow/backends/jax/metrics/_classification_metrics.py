from abc import ABC, abstractmethod

import numpy as np
from jax.typing import ArrayLike
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from sc_flow.backends.jax.metrics.__types import ClassificationState

__all__ = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1Score",
    "AUROC",
    "AveragePrecision",
    "CrossEntropyLoss",
]


class _ClassificationMetric(ABC):
    """Base class providing shared init/update/reset for all classification metrics."""

    def init(self) -> ClassificationState:
        return ClassificationState(preds=[], targets=[])

    def update(self, state: ClassificationState, preds: ArrayLike, targets: ArrayLike) -> ClassificationState:
        return ClassificationState(
            preds=state.preds + [np.array(preds)],
            targets=state.targets + [np.array(targets)],
        )

    @abstractmethod
    def compute(self, state: ClassificationState) -> float: ...

    def reset(self) -> ClassificationState:
        return self.init()

    def _concat(self, state: ClassificationState) -> tuple[np.ndarray, np.ndarray]:
        return np.concatenate(state.preds), np.concatenate(state.targets)


# ---------------------------------------------------------------------------
# Threshold-based metrics — preds should be binary {0, 1}
# ---------------------------------------------------------------------------


class Accuracy(_ClassificationMetric):
    def compute(self, state: ClassificationState) -> float:
        preds, targets = self._concat(state)
        return float(accuracy_score(targets, preds))


class Precision(_ClassificationMetric):
    def compute(self, state: ClassificationState) -> float:
        preds, targets = self._concat(state)
        return float(precision_score(targets, preds, zero_division=0))


class Recall(_ClassificationMetric):
    def compute(self, state: ClassificationState) -> float:
        preds, targets = self._concat(state)
        return float(recall_score(targets, preds, zero_division=0))


class F1Score(_ClassificationMetric):
    def compute(self, state: ClassificationState) -> float:
        preds, targets = self._concat(state)
        return float(f1_score(targets, preds, zero_division=0))


# ---------------------------------------------------------------------------
# Probability-based metrics — preds should be soft probabilities in [0, 1]
# ---------------------------------------------------------------------------


class AUROC(_ClassificationMetric):
    def compute(self, state: ClassificationState) -> float:
        probs, targets = self._concat(state)
        return float(roc_auc_score(targets, probs))


class AveragePrecision(_ClassificationMetric):
    def compute(self, state: ClassificationState) -> float:
        probs, targets = self._concat(state)
        return float(average_precision_score(targets, probs))


class CrossEntropyLoss(_ClassificationMetric):
    def compute(self, state: ClassificationState) -> float:
        probs, targets = self._concat(state)
        return float(log_loss(targets, probs))
