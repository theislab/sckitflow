import jax
import jax.numpy as jnp
import numpy as np
import pytest
import optax
from flax.training import train_state

from sc_flow.backends.jax.methods import BaseMethod

class DummyVFState(train_state.TrainState):
    pass


class TestVF:
    def __init__(self):
        # simulate model parameters as a scalar
        self.params = {"weight": jnp.array(1.0)}

    def apply_fn(self, variables, t, x_t, conditions, rngs=None):
        # a trivial vector field: v_t = weight * x_t
        w = variables["params"]["weight"]
        return w * x_t

    def create_train_state(self, input_dim: int):
        tx = optax.identity()  # no-op optimizer, does nothing
        return DummyVFState.create(
            apply_fn=self.apply_fn,
            params=self.params,
            tx=tx,
        )

class DummyProbabilityPath:
    def compute_xt(self, rng, t, source, target):
        # linear interpolation: x_t = (1 - t) * source + t * target
        t = t.reshape(-1, 1)
        return (1.0 - t) * source + t * target

    def compute_ut(self, t, x_t, source, target):
        # simple ground truth vector field (difference)
        return target - source


def dummy_match_fn(src, tgt):
    n = src.shape[0]
    m = tgt.shape[0]
    tmat = jnp.ones((n, m))
    return tmat / (n*m)

def dummy_time_sampler(rng, n):
    return jax.random.uniform(rng, shape=(n,), minval=0.0, maxval=1.0)


def test_step_fn_runs_without_error():
    vf = TestVF()
    probability_path = DummyProbabilityPath()
    match_fn = dummy_match_fn
    time_sampler = dummy_time_sampler
    target_dim = 3

    method = BaseMethod(
        vf=vf,
        probability_path=probability_path,
        time_sampler=time_sampler,
        match_fn=match_fn,
        target_dim=target_dim,
    )

    # Create dummy batch
    batch_size, dim = 4, 3
    batch = {
        "src_cell_data": jnp.ones((batch_size, dim)),
        "tgt_cell_data": jnp.ones((batch_size, dim)) * 2,
        "condition": {"dummy_cond": jnp.ones((batch_size, 1))},
    }

    rng = jax.random.PRNGKey(42)
    loss = method.step_fn(rng, batch)

    assert jnp.isscalar(loss) or np.isscalar(loss)
    assert jnp.isfinite(loss)
