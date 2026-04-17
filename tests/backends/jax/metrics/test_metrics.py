import jax.numpy as jnp
import numpy as np
from sklearn.metrics import pairwise_distances

from sc_flow.backends.jax.metrics import (
    EnergyDistance,
    MaximumMeanDiscrepancy,
    MaximumMeanDiscrepancyState,
    SinkhornDivergence,
)
from sc_flow.backends.jax.metrics._utils import compute_e_distance


def _compute_mmd_sklearn(pred: np.ndarray, target: np.ndarray, gamma: float) -> float:
    """Reference implementation using sklearn pairwise distances."""

    def rbf_kernel(x1, x2, gamma):
        sq_dists = pairwise_distances(x1, x2, metric="sqeuclidean")
        return np.exp(-gamma * sq_dists)

    xx = rbf_kernel(pred, pred, gamma=gamma).mean()
    yy = rbf_kernel(target, target, gamma=gamma).mean()
    xy = rbf_kernel(pred, target, gamma=gamma).mean()
    return xx + yy - 2 * xy


def test_energy_distance_vs_reference():
    """Test that EnergyDistance matches compute_e_distance from _utils."""
    np.random.seed(42)
    n_samples, n_features = 100, 50

    pred = jnp.array(np.random.randn(n_samples, n_features).astype(np.float32))
    target = jnp.array(np.random.randn(n_samples, n_features).astype(np.float32))

    metric = EnergyDistance()
    state = metric.init()
    state = metric.update(state, pred, target)
    result = float(metric.compute(state))

    expected = float(compute_e_distance(pred, target))

    assert np.isclose(result, expected, rtol=1e-3, atol=1e-5), (
        f"JAX result {result} does not match reference result {expected}"
    )


def test_energy_distance_accumulates_correctly():
    """Test that multiple update calls accumulate and compute the average."""
    np.random.seed(0)
    n_samples, n_features = 50, 20

    metric = EnergyDistance()
    state = metric.init()

    batch1_pred = jnp.array(np.random.randn(n_samples, n_features).astype(np.float32))
    batch1_target = jnp.array(np.random.randn(n_samples, n_features).astype(np.float32))
    batch2_pred = jnp.array(np.random.randn(n_samples, n_features).astype(np.float32))
    batch2_target = jnp.array(np.random.randn(n_samples, n_features).astype(np.float32))

    state = metric.update(state, batch1_pred, batch1_target)
    state = metric.update(state, batch2_pred, batch2_target)

    result = float(metric.compute(state))

    e1 = float(compute_e_distance(batch1_pred, batch1_target))
    e2 = float(compute_e_distance(batch2_pred, batch2_target))
    expected = (e1 + e2) / 2.0

    assert np.isclose(result, expected, rtol=1e-3, atol=1e-5), (
        f"Accumulated result {result} does not match expected average {expected}"
    )


def test_energy_distance_reset():
    """Test that reset returns the metric to its initial state."""
    np.random.seed(1)
    n_samples, n_features = 30, 10

    pred = jnp.array(np.random.randn(n_samples, n_features).astype(np.float32))
    target = jnp.array(np.random.randn(n_samples, n_features).astype(np.float32))

    metric = EnergyDistance()
    state = metric.init()
    state = metric.update(state, pred, target)

    reset_state = metric.reset()
    assert float(reset_state.total) == 0.0
    assert float(reset_state.energy_distance_raw) == 0.0


def test_maximum_mean_discrepancy_vs_sklearn():
    """Test that MaximumMeanDiscrepancy matches sklearn reference for a single gamma."""
    np.random.seed(42)
    n_samples, n_features = 100, 50

    pred = jnp.array(np.random.randn(n_samples, n_features).astype(np.float32))
    target = jnp.array(np.random.randn(n_samples, n_features).astype(np.float32))

    gamma = 1.0
    metric = MaximumMeanDiscrepancy(gammas=[gamma])
    state = MaximumMeanDiscrepancyState(mmd_raw=jnp.zeros(()), total=jnp.zeros(()))
    state = metric.update(state, pred, target)
    result = float(metric.compute(state))

    result_sklearn = _compute_mmd_sklearn(np.array(pred), np.array(target), gamma=gamma)

    assert np.isclose(result, result_sklearn, rtol=1e-3, atol=1e-5), (
        f"JAX MMD result {result} does not match sklearn result {result_sklearn}"
    )


def test_maximum_mean_discrepancy_multiple_gammas():
    """Test that MMD with multiple gammas averages over them correctly."""
    np.random.seed(7)
    n_samples, n_features = 80, 30

    pred = jnp.array(np.random.randn(n_samples, n_features).astype(np.float32))
    target = jnp.array(np.random.randn(n_samples, n_features).astype(np.float32))

    gammas = [0.5, 1.0, 2.0]
    metric = MaximumMeanDiscrepancy(gammas=gammas)
    state = MaximumMeanDiscrepancyState(mmd_raw=jnp.zeros(()), total=jnp.zeros(()))
    state = metric.update(state, pred, target)
    result = float(metric.compute(state))

    per_gamma = [_compute_mmd_sklearn(np.array(pred), np.array(target), g) for g in gammas]
    expected = float(np.mean(per_gamma))

    assert np.isclose(result, expected, rtol=1e-3, atol=1e-5), (
        f"Multi-gamma MMD result {result} does not match expected {expected}"
    )


def test_maximum_mean_discrepancy_accumulates_correctly():
    """Test that multiple MMD update calls accumulate and compute the average."""
    np.random.seed(3)
    n_samples, n_features = 40, 15

    batch1_pred = jnp.array(np.random.randn(n_samples, n_features).astype(np.float32))
    batch1_target = jnp.array(np.random.randn(n_samples, n_features).astype(np.float32))
    batch2_pred = jnp.array(np.random.randn(n_samples, n_features).astype(np.float32))
    batch2_target = jnp.array(np.random.randn(n_samples, n_features).astype(np.float32))

    gamma = 1.0
    metric = MaximumMeanDiscrepancy(gammas=[gamma])
    state = MaximumMeanDiscrepancyState(mmd_raw=jnp.zeros(()), total=jnp.zeros(()))
    state = metric.update(state, batch1_pred, batch1_target)
    state = metric.update(state, batch2_pred, batch2_target)
    result = float(metric.compute(state))

    m1 = _compute_mmd_sklearn(np.array(batch1_pred), np.array(batch1_target), gamma)
    m2 = _compute_mmd_sklearn(np.array(batch2_pred), np.array(batch2_target), gamma)
    expected = (m1 + m2) / 2.0

    assert np.isclose(result, expected, rtol=1e-3, atol=1e-5), (
        f"Accumulated MMD result {result} does not match expected {expected}"
    )


def test_sinkhorn_divergence_non_negative():
    """Test that SinkhornDivergence returns a non-negative value."""
    np.random.seed(42)
    n_samples, n_features = 50, 10

    pred = jnp.array(np.random.randn(n_samples, n_features).astype(np.float32))
    target = jnp.array(np.random.randn(n_samples, n_features).astype(np.float32))

    metric = SinkhornDivergence(epsilon=1e-2)
    state = metric.init()
    state = metric.update(state, pred, target)
    result = metric.compute(state)

    assert result >= 0.0, f"SinkhornDivergence should be non-negative, got {result}"


def test_sinkhorn_divergence_identical_distributions():
    """Test that SinkhornDivergence is near zero for identical distributions."""
    np.random.seed(42)
    n_samples, n_features = 50, 10

    pred = jnp.array(np.random.randn(n_samples, n_features).astype(np.float32))

    metric = SinkhornDivergence(epsilon=1e-2)
    state = metric.init()
    state = metric.update(state, pred, pred)
    result = metric.compute(state)

    assert abs(result) < 1e-3, f"SinkhornDivergence between identical samples should be near zero, got {result}"


def test_sinkhorn_divergence_accumulates_correctly():
    """Test that multiple SinkhornDivergence update calls average correctly."""
    np.random.seed(5)
    n_samples, n_features = 30, 8

    batch1_pred = jnp.array(np.random.randn(n_samples, n_features).astype(np.float32))
    batch1_target = jnp.array(np.random.randn(n_samples, n_features).astype(np.float32))
    batch2_pred = jnp.array(np.random.randn(n_samples, n_features).astype(np.float32))
    batch2_target = jnp.array(np.random.randn(n_samples, n_features).astype(np.float32))

    metric = SinkhornDivergence(epsilon=1e-2)
    state = metric.init()
    state = metric.update(state, batch1_pred, batch1_target)
    state = metric.update(state, batch2_pred, batch2_target)
    result = metric.compute(state)

    # Compute each batch individually and verify the average
    state1 = metric.init()
    state1 = metric.update(state1, batch1_pred, batch1_target)
    r1 = metric.compute(state1)

    state2 = metric.init()
    state2 = metric.update(state2, batch2_pred, batch2_target)
    r2 = metric.compute(state2)

    expected = (r1 + r2) / 2.0
    assert np.isclose(result, expected, rtol=1e-3, atol=1e-5), (
        f"Accumulated Sinkhorn result {result} does not match expected average {expected}"
    )


def test_sinkhorn_divergence_reset():
    """Test that reset returns the SinkhornDivergence to its initial state."""
    np.random.seed(2)
    n_samples, n_features = 20, 5

    pred = jnp.array(np.random.randn(n_samples, n_features).astype(np.float32))
    target = jnp.array(np.random.randn(n_samples, n_features).astype(np.float32))

    metric = SinkhornDivergence(epsilon=1e-2)
    state = metric.init()
    state = metric.update(state, pred, target)

    reset_state = metric.reset()
    assert float(reset_state.total) == 0.0
    assert float(reset_state.sinkhorn_raw) == 0.0
