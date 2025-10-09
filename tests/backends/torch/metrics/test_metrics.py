import numpy as np
import torch
from sklearn.metrics import pairwise_distances

from sc_flow.backends.torch.metrics._metrics import EnergyDistance, MaximumMeanDiscrepancy


def _compute_energy_distance_sklearn(pred: np.ndarray, target: np.ndarray) -> float:
    """Reference implementation using sklearn pairwise distances."""
    # Energy distance formula: 2 * E[||X-Y||] - E[||X-X'||] - E[||Y-Y'||]
    pred_pred_dist = pairwise_distances(pred, pred, metric='euclidean')
    target_target_dist = pairwise_distances(target, target, metric='euclidean')
    pred_target_dist = pairwise_distances(pred, target, metric='euclidean')

    sigma_pred = pred_pred_dist.mean()
    sigma_target = target_target_dist.mean()
    delta = pred_target_dist.mean()

    return 2. * delta - sigma_pred - sigma_target


def _compute_maximum_mean_discrepancy_sklearn(pred: np.ndarray, target: np.ndarray, gamma: float) -> float:
    """Reference implementation using sklearn pairwise distances."""
    def rbf_kernel(x1, x2, gamma):
        sq_dists = pairwise_distances(x1, x2, metric='sqeuclidean')
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

    # Compare results (use relative tolerance due to floating point arithmetic)
    assert np.isclose(
        result,
        result_sklean,
        rtol=1e-3,
        atol=1e-5
    ), f"Batched result {result} does not match sklearn result {result_sklean}"


def test_maximum_mean_discrepancy_vs_sklearn():
    """Test that batched MaximumMeanDiscrepancy matches sklearn reference on concatenated data."""
    n_samples = 100
    n_features = 50

    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    metric = MaximumMeanDiscrepancy(gammas=[1.])
    pred = torch.randn(n_samples, n_features)
    target = torch.randn(n_samples, n_features)
    metric.update(pred, target)

    result = metric.compute().item()
    result_sklearn = _compute_maximum_mean_discrepancy_sklearn(pred.numpy(), target.numpy(), gamma=1.)

    assert np.isclose(
        result,
        result_sklearn,
        rtol=1e-3,
        atol=1e-5
    ), f"Batched result {result} does not match sklearn result {result_sklearn}"
