import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from sc_flow.backends.jax.metrics import (
    AUROC,
    Accuracy,
    AveragePrecision,
    CrossEntropyLoss,
    F1Score,
    Precision,
    Recall,
)


def _binary_data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    targets = rng.integers(0, 2, size=n)
    probs = rng.uniform(0.0, 1.0, size=n)
    preds = (probs >= 0.5).astype(int)
    return preds, probs, targets


def test_accuracy_vs_sklearn():
    preds, _, targets = _binary_data(seed=1)
    metric = Accuracy()
    state = metric.init()
    state = metric.update(state, preds, targets)
    assert np.isclose(metric.compute(state), accuracy_score(targets, preds))


def test_accuracy_accumulates():
    """Batched updates should concatenate before computing."""
    preds1, _, targets1 = _binary_data(n=100, seed=2)
    preds2, _, targets2 = _binary_data(n=80, seed=3)

    metric = Accuracy()
    state = metric.init()
    state = metric.update(state, preds1, targets1)
    state = metric.update(state, preds2, targets2)

    expected = accuracy_score(
        np.concatenate([targets1, targets2]),
        np.concatenate([preds1, preds2]),
    )
    assert np.isclose(metric.compute(state), expected)


def test_accuracy_reset():
    preds, _, targets = _binary_data(seed=4)
    metric = Accuracy()
    state = metric.init()
    state = metric.update(state, preds, targets)
    reset_state = metric.reset()
    assert reset_state.preds == [] and reset_state.targets == []


def test_precision_vs_sklearn():
    preds, _, targets = _binary_data(seed=5)
    metric = Precision()
    state = metric.init()
    state = metric.update(state, preds, targets)
    assert np.isclose(metric.compute(state), precision_score(targets, preds, zero_division=0))


def test_precision_accumulates():
    preds1, _, targets1 = _binary_data(n=100, seed=6)
    preds2, _, targets2 = _binary_data(n=80, seed=7)

    metric = Precision()
    state = metric.init()
    state = metric.update(state, preds1, targets1)
    state = metric.update(state, preds2, targets2)

    expected = precision_score(
        np.concatenate([targets1, targets2]),
        np.concatenate([preds1, preds2]),
        zero_division=0,
    )
    assert np.isclose(metric.compute(state), expected)


def test_recall_vs_sklearn():
    preds, _, targets = _binary_data(seed=8)
    metric = Recall()
    state = metric.init()
    state = metric.update(state, preds, targets)
    assert np.isclose(metric.compute(state), recall_score(targets, preds, zero_division=0))


def test_recall_accumulates():
    preds1, _, targets1 = _binary_data(n=100, seed=9)
    preds2, _, targets2 = _binary_data(n=80, seed=10)

    metric = Recall()
    state = metric.init()
    state = metric.update(state, preds1, targets1)
    state = metric.update(state, preds2, targets2)

    expected = recall_score(
        np.concatenate([targets1, targets2]),
        np.concatenate([preds1, preds2]),
        zero_division=0,
    )
    assert np.isclose(metric.compute(state), expected)


def test_f1_vs_sklearn():
    preds, _, targets = _binary_data(seed=11)
    metric = F1Score()
    state = metric.init()
    state = metric.update(state, preds, targets)
    assert np.isclose(metric.compute(state), f1_score(targets, preds, zero_division=0))


def test_f1_accumulates():
    preds1, _, targets1 = _binary_data(n=100, seed=12)
    preds2, _, targets2 = _binary_data(n=80, seed=13)

    metric = F1Score()
    state = metric.init()
    state = metric.update(state, preds1, targets1)
    state = metric.update(state, preds2, targets2)

    expected = f1_score(
        np.concatenate([targets1, targets2]),
        np.concatenate([preds1, preds2]),
        zero_division=0,
    )
    assert np.isclose(metric.compute(state), expected)


def test_auroc_vs_sklearn():
    _, probs, targets = _binary_data(seed=14)
    metric = AUROC()
    state = metric.init()
    state = metric.update(state, probs, targets)
    assert np.isclose(metric.compute(state), roc_auc_score(targets, probs))


def test_auroc_accumulates():
    _, probs1, targets1 = _binary_data(n=100, seed=15)
    _, probs2, targets2 = _binary_data(n=80, seed=16)

    metric = AUROC()
    state = metric.init()
    state = metric.update(state, probs1, targets1)
    state = metric.update(state, probs2, targets2)

    expected = roc_auc_score(
        np.concatenate([targets1, targets2]),
        np.concatenate([probs1, probs2]),
    )
    assert np.isclose(metric.compute(state), expected)


def test_average_precision_vs_sklearn():
    _, probs, targets = _binary_data(seed=17)
    metric = AveragePrecision()
    state = metric.init()
    state = metric.update(state, probs, targets)
    assert np.isclose(metric.compute(state), average_precision_score(targets, probs))


def test_average_precision_accumulates():
    _, probs1, targets1 = _binary_data(n=100, seed=18)
    _, probs2, targets2 = _binary_data(n=80, seed=19)

    metric = AveragePrecision()
    state = metric.init()
    state = metric.update(state, probs1, targets1)
    state = metric.update(state, probs2, targets2)

    expected = average_precision_score(
        np.concatenate([targets1, targets2]),
        np.concatenate([probs1, probs2]),
    )
    assert np.isclose(metric.compute(state), expected)


def test_cross_entropy_vs_sklearn():
    _, probs, targets = _binary_data(seed=20)
    metric = CrossEntropyLoss()
    state = metric.init()
    state = metric.update(state, probs, targets)
    assert np.isclose(metric.compute(state), log_loss(targets, probs))


def test_cross_entropy_accumulates():
    _, probs1, targets1 = _binary_data(n=100, seed=21)
    _, probs2, targets2 = _binary_data(n=80, seed=22)

    metric = CrossEntropyLoss()
    state = metric.init()
    state = metric.update(state, probs1, targets1)
    state = metric.update(state, probs2, targets2)

    expected = log_loss(
        np.concatenate([targets1, targets2]),
        np.concatenate([probs1, probs2]),
    )
    assert np.isclose(metric.compute(state), expected)
