import jax
import jax.numpy as jnp
from chex import dataclass

__all__ = ["StepData", "ema_update"]


@jax.jit
def ema_update(current_model_params: dict, new_model_params: dict, ema: float) -> dict:
    """
    Update parameters using exponential moving average.

    Parameters
    ----------
        current_model_parames
            Current parameters.
        new_model_params
            New parameters to be averaged.
        ema
            Exponential moving average factor
            between `0` and `1`. `0` means no update, `1` means full update.

    Returns
    -------
        Updated parameters after applying EMA.
    """
    new_inference_model_params = jax.tree.map(
        lambda p, tp: p * (1 - ema) + tp * ema, current_model_params, new_model_params
    )
    return new_inference_model_params


def _multivariate_normal(
    rng: jax.Array,
    shape: tuple[int, ...],
    dim: int,
    mean: float = 0.0,
    cov: float = 1.0,
) -> jnp.ndarray:
    mean = jnp.full(dim, fill_value=mean)
    cov = jnp.diag(jnp.full(dim, fill_value=cov))
    return jax.random.multivariate_normal(rng, mean=mean, cov=cov, shape=shape)


def default_prng_key(rng: jax.Array | None) -> jax.Array:
    """Get the default PRNG key.

    Parameters
    ----------
    rng: PRNG key.

    Returns
    -------
      If ``rng = None``, returns the default PRNG key. Otherwise, it returns
      the unmodified ``rng`` key.
    """
    return jax.random.key(0) if rng is None else rng


@dataclass
class StepData:
    target_state: jnp.ndarray
    target_coupling_lin: jnp.ndarray
    target_coupling_quad: jnp.ndarray | None
    target_condition_data: jnp.ndarray | None
    target_group_data: jnp.ndarray | None
    source_state: jnp.ndarray | None
    source_coupling_lin: jnp.ndarray | None
    source_coupling_quad: jnp.ndarray | None
    source_condition_data: jnp.ndarray | None
    source_group_data: jnp.ndarray | None
