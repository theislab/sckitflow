import jax

from sc_flow.backends.jax._types import ArrayLike


def pairwise_squeuclidean(x: ArrayLike, y: ArrayLike) -> ArrayLike:
    """Compute pairwise squared euclidean distances."""
    return ((x[:, None, :] - y[None, :, :]) ** 2).sum(-1)


def compute_e_distance(x: ArrayLike, y: ArrayLike) -> float:
    """Compute the energy distance between x and y as in :cite:`Peidli2024`.

    Parameters
    ----------
        x
            An array of shape [num_samples, num_features].
        y
            An array of shape [num_samples, num_features].

    Returns
    -------
        A scalar denoting the energy distance value.
    """
    sigma_X = pairwise_squeuclidean(x, x).mean()
    sigma_Y = pairwise_squeuclidean(y, y).mean()
    delta = pairwise_squeuclidean(x, y).mean()
    return 2 * delta - sigma_X - sigma_Y


@jax.jit
def compute_e_distance_fast(x: ArrayLike, y: ArrayLike) -> float:
    """Compute the energy distance between x and y as in :cite:`Peidli2024`.

    Parameters
    ----------
        x
            An array of shape [num_samples, num_features].
        y
            An array of shape [num_samples, num_features].

    Returns
    -------
        A scalar denoting the energy distance value.
    """
    return compute_e_distance(x, y)
