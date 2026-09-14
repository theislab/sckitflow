import pytest
import torch

from sckitflow.core.methods.inference._utils import aggregate_predictions


# -------------------- 3-dim latent: no trajectory --------------------
def test_aggregate_3d_no_trajectory():
    N, B, D = 3, 4, 5
    latent_shape = torch.Size([N, B, D])
    predictions = torch.randn(N, B, D)

    X, traj, raw_samples = aggregate_predictions(predictions, latent_shape, return_trajectory=False)

    assert traj is None
    assert X.shape == (B, D)
    assert raw_samples.shape == (N, B, D)
    torch.testing.assert_close(X, predictions.mean(dim=0))
    torch.testing.assert_close(raw_samples, predictions)


# -------------------- 3-dim latent: with trajectory --------------------
def test_aggregate_3d_with_trajectory():
    N, B, D = 3, 4, 5
    T = 10
    latent_shape = torch.Size([N, B, D])
    predictions = torch.randn(T, N, B, D)

    X, traj, raw_samples = aggregate_predictions(predictions, latent_shape, return_trajectory=True)

    assert traj.shape == (T, N, B, D)
    assert X.shape == (B, D)
    assert raw_samples.shape == (N, B, D)
    torch.testing.assert_close(traj, predictions)
    torch.testing.assert_close(X, predictions[-1].mean(dim=0))
    torch.testing.assert_close(raw_samples, predictions[-1])


# -------------------- 2-dim latent: no trajectory --------------------
def test_aggregate_2d_no_trajectory():
    B, D = 4, 5
    latent_shape = torch.Size([B, D])
    predictions = torch.randn(B, D)

    X, traj, raw_samples = aggregate_predictions(predictions, latent_shape, return_trajectory=False)

    assert traj is None
    assert raw_samples is None
    assert X.shape == (B, D)
    torch.testing.assert_close(X, predictions)


# -------------------- 2-dim latent: with trajectory --------------------
def test_aggregate_2d_with_trajectory():
    B, D = 4, 5
    T = 10
    latent_shape = torch.Size([B, D])
    predictions = torch.randn(T, B, D)

    X, traj, raw_samples = aggregate_predictions(predictions, latent_shape, return_trajectory=True)

    assert raw_samples is None
    assert X.shape == (B, D)
    assert traj.shape == (T, 1, B, D)
    torch.testing.assert_close(X, predictions[-1])
    torch.testing.assert_close(traj, predictions.unsqueeze(1))


# -------------------- Averaging correctness (known values) --------------------
def test_aggregate_3d_average_is_exact_mean():
    """Averaging must match ``Tensor.mean(dim=0)`` on exact values."""
    N, B, D = 4, 2, 3
    latent_shape = torch.Size([N, B, D])
    # Distinct rows so an incorrect mean would be obvious.
    predictions = torch.arange(N * B * D, dtype=torch.float32).reshape(N, B, D)

    X, _, raw_samples = aggregate_predictions(predictions, latent_shape, return_trajectory=False)

    expected_mean = predictions.mean(dim=0)
    torch.testing.assert_close(X, expected_mean)
    torch.testing.assert_close(raw_samples, predictions)


def test_aggregate_3d_trajectory_final_matches_last_step():
    """When a trajectory is returned, ``X`` must be the average of the last step."""
    N, B, D = 3, 2, 4
    T = 6
    latent_shape = torch.Size([N, B, D])
    predictions = torch.randn(T, N, B, D)

    X, traj, raw_samples = aggregate_predictions(predictions, latent_shape, return_trajectory=True)

    torch.testing.assert_close(X, traj[-1].mean(dim=0))
    torch.testing.assert_close(raw_samples, traj[-1])


# -------------------- Edge cases: N = 1 --------------------
def test_aggregate_3d_single_sample_returns_mean_equal_to_sample():
    N, B, D = 1, 4, 5
    latent_shape = torch.Size([N, B, D])
    predictions = torch.randn(N, B, D)

    X, _, raw_samples = aggregate_predictions(predictions, latent_shape, return_trajectory=False)

    assert X.shape == (B, D)
    torch.testing.assert_close(X, predictions[0])
    torch.testing.assert_close(raw_samples, predictions)


# -------------------- dtype preservation --------------------
def test_aggregate_preserves_dtype():
    N, B, D = 3, 2, 4
    latent_shape = torch.Size([N, B, D])
    predictions = torch.randn(N, B, D, dtype=torch.float64)

    X, _, raw_samples = aggregate_predictions(predictions, latent_shape, return_trajectory=False)

    assert X.dtype == torch.float64
    assert raw_samples.dtype == torch.float64


# -------------------- device preservation --------------------
def test_aggregate_preserves_device():
    N, B, D = 3, 2, 4
    latent_shape = torch.Size([N, B, D])
    predictions = torch.randn(N, B, D, device="cpu")

    X, _, raw_samples = aggregate_predictions(predictions, latent_shape, return_trajectory=False)

    assert X.device == torch.device("cpu")
    assert raw_samples.device == torch.device("cpu")


# -------------------- Invalid latent shapes --------------------
@pytest.mark.parametrize(
    "latent_shape",
    [
        torch.Size([]),  # scalar
        torch.Size([5]),  # 1-dim
        torch.Size([2, 3, 4, 5]),  # 4-dim
    ],
)
def test_aggregate_rejects_invalid_latent_shape(latent_shape):
    # The function validates latent_shape before inspecting predictions,
    # so predictions can be any placeholder tensor.
    predictions = torch.empty(0)
    with pytest.raises(ValueError, match="Invalid shape for source latent state"):
        aggregate_predictions(predictions, latent_shape, return_trajectory=False)
