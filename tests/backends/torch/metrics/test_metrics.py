import numpy as np
import pytest
import torch
from sklearn.metrics import pairwise_distances

pytest.importorskip("torchmetrics")

from sklearn.metrics import r2_score

from sc_flow.core.metrics._metrics import EnergyDistance, MaximumMeanDiscrepancy, RSquared


def _compute_energy_distance_sklearn(pred: np.ndarray, target: np.ndarray) -> float:
    """Reference: cellflow's scPerturb E-distance with **squared-Euclidean** costs (Peidli2024)."""
    # E = 2 * E[||X-Y||^2] - E[||X-X'||^2] - E[||Y-Y'||^2]   (sqeuclidean, matching cellflow)
    sigma_pred = pairwise_distances(pred, pred, metric="sqeuclidean").mean()
    sigma_target = pairwise_distances(target, target, metric="sqeuclidean").mean()
    delta = pairwise_distances(pred, target, metric="sqeuclidean").mean()
    return 2.0 * delta - sigma_pred - sigma_target


def _cellflow_pairwise_sqeuclidean(x, y):
    """cellflow's exact ``pairwise_squeuclidean`` (broadcast form)."""
    return ((x[:, None, :] - y[None, :, :]) ** 2).sum(-1)


def _compute_energy_distance_cellflow(x: np.ndarray, y: np.ndarray) -> float:
    """cellflow ``compute_e_distance`` verbatim (x=pred, y=target)."""
    sigma_x = _cellflow_pairwise_sqeuclidean(x, x).mean()
    sigma_y = _cellflow_pairwise_sqeuclidean(y, y).mean()
    delta = _cellflow_pairwise_sqeuclidean(x, y).mean()
    return 2 * delta - sigma_x - sigma_y


def _compute_maximum_mean_discrepancy_sklearn(pred: np.ndarray, target: np.ndarray, gamma: float) -> float:
    """Reference implementation using sklearn pairwise distances."""

    def rbf_kernel(x1, x2, gamma):
        sq_dists = pairwise_distances(x1, x2, metric="sqeuclidean")
        return np.exp(-gamma * sq_dists)

    xx = rbf_kernel(pred, pred, gamma=gamma).mean()
    yy = rbf_kernel(target, target, gamma=gamma).mean()
    xy = rbf_kernel(pred, target, gamma=gamma).mean()
    return xx + yy - 2 * xy


def test_energy_distance_vs_sklearn():
    """Test that batched EnergyDistance matches sklearn reference on concatenated data."""
    n_samples = 100
    n_features = 50
    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    metric = EnergyDistance()
    pred = torch.randn(n_samples, n_features)
    target = torch.randn(n_samples, n_features)
    metric.update(pred, target)
    result = metric.compute().item()
    result_sklean = _compute_energy_distance_sklearn(pred.numpy(), target.numpy())
    result_cellflow = _compute_energy_distance_cellflow(pred.numpy(), target.numpy())

    # Compare results (use relative tolerance due to floating point arithmetic)
    assert np.isclose(result, result_sklean, rtol=1e-3, atol=1e-5), (
        f"Batched result {result} does not match sklearn result {result_sklean}"
    )
    # And match cellflow's compute_e_distance formula exactly (squared-Euclidean).
    assert np.isclose(result, result_cellflow, rtol=1e-3, atol=1e-5), (
        f"Result {result} does not match cellflow e_distance {result_cellflow}"
    )


def test_maximum_mean_discrepancy_vs_sklearn():
    """Test that batched MaximumMeanDiscrepancy matches sklearn reference on concatenated data."""
    n_samples = 100
    n_features = 50

    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    metric = MaximumMeanDiscrepancy(gammas=[1.0])
    pred = torch.randn(n_samples, n_features)
    target = torch.randn(n_samples, n_features)
    metric.update(pred, target)

    result = metric.compute().item()
    result_sklearn = _compute_maximum_mean_discrepancy_sklearn(pred.numpy(), target.numpy(), gamma=1.0)

    assert np.isclose(result, result_sklearn, rtol=1e-3, atol=1e-5), (
        f"Batched result {result} does not match sklearn result {result_sklearn}"
    )


def test_r_squared_matches_cellflow_r2_score():
    """RSquared must equal cellflow's compute_r_squared = r2_score(mean(true), mean(pred))."""
    torch.manual_seed(0)
    # Different cell counts (populations, not paired rows) — the metric reduces each to its feature mean.
    pred = torch.randn(80, 32)
    target = torch.randn(120, 32) * 1.5 + 0.3

    metric = RSquared()
    metric.update(pred, target)
    result = metric.compute().item()

    # cellflow: compute_r_squared(x=true, y=pred) = r2_score(mean(true, 0), mean(pred, 0))
    expected = r2_score(target.numpy().mean(axis=0), pred.numpy().mean(axis=0))
    assert np.isclose(result, expected, rtol=1e-4, atol=1e-5), (
        f"RSquared {result} != cellflow r2_score {expected}"
    )


def test_r_squared_averages_over_conditions():
    """compute() averages R² over the conditions fed via successive update() calls."""
    torch.manual_seed(1)
    metric = RSquared()
    expected = []
    for _ in range(3):
        pred, target = torch.randn(50, 16), torch.randn(70, 16) * 2 - 1
        metric.update(pred, target)
        expected.append(r2_score(target.numpy().mean(0), pred.numpy().mean(0)))
    assert np.isclose(metric.compute().item(), float(np.mean(expected)), rtol=1e-4, atol=1e-5)
