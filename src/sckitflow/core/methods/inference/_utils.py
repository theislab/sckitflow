import torch

__all__ = ["aggregate_predictions"]


def aggregate_predictions(
    predictions: torch.Tensor,
    latent_shape: torch.Size,
    return_trajectory: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """
    Convert solver output into final X, traj, and raw_samples.

    Parameters
    ----------
    predictions : torch.Tensor
        Solver output from ``ODESolver``. If ``return_trajectory=True``, the shape is
        ``(num_steps, *latent_shape)``. Otherwise, the shape is ``latent_shape``.
    latent_shape : torch.Size
        Original shape of the latent tensor (for example,
        ``(n_samples, batch_size, dim)`` or ``(batch_size, dim)``).
    return_trajectory : bool
        Whether the solver returned a full trajectory.

    Returns
    -------
    X : torch.Tensor
        Final predicted state. If ``latent_shape`` includes a sample dimension,
        this is the mean of the final states over samples with shape
        ``(batch_size, dim)``. Otherwise, this is the final state tensor with
        shape ``(batch_size, dim)``.
    traj : torch.Tensor or None
        Full trajectory tensor, not averaged over samples. When a sample
        dimension is present, the shape is ``(num_steps, n_samples, batch_size, dim)``.
        Otherwise, the returned tensor includes a singleton sample dimension
        and has shape ``(num_steps, 1, batch_size, dim)``. Returns ``None`` if
        ``return_trajectory=False``.
    raw_samples : torch.Tensor or None
        Raw final states for each sample with shape ``(n_samples, batch_size, dim)``
        when a sample dimension is present, or ``None`` otherwise.
    """
    # ---- Determine if we have a sample dimension from latent state ----
    n_latent_dims = len(latent_shape)
    if n_latent_dims == 3:
        n_samples = latent_shape[0]
        batch_size = latent_shape[1]
        has_sample_dim = average_samples = True
    elif n_latent_dims == 2:
        has_sample_dim = average_samples = False
    else:
        raise ValueError(f"Invalid shape for source latent state: {latent_shape}")

    # ---- Reshape output when needed ----
    if has_sample_dim:
        # predictions will be of shape (T, N, B, D)
        n_steps = predictions.shape[0]
        # ---- Handle optional trajectory ----
        if return_trajectory:
            traj_full = predictions.reshape(n_steps, n_samples, batch_size, -1)
            all_final = traj_full[-1]  # (N, B, D)
        else:
            all_final = predictions.reshape(n_samples, batch_size, -1)
            traj_full = None
    else:
        # ---- Handle optional trajectory ----
        if return_trajectory:
            all_final = predictions[-1]  # (B, D)
            traj_full = predictions.unsqueeze(1)  # unsqueeze sample dim (N=1) (T, 1, B, D)
        else:
            all_final = predictions
            traj_full = None

    # ---- Average when generating multiple samples----
    if average_samples:
        X = all_final.mean(dim=0)  # (B, D)
        raw_samples = all_final  # (N, B, D)
    else:
        X = all_final  # (B, D)
        raw_samples = None

    return X, traj_full, raw_samples
