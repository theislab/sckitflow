import jax
import jax.numpy as jnp
import numpy as np
import pytest
import optax
import functools
from flax.training import train_state

from sc_flow.backends.jax.methods import BaseMethod
from sc_flow.backends.jax.nn._vf import MLPUnconditionalVF
from sc_flow.backends.jax.probability_paths import LinearDiracProbabilityPath
class DummyVFState(train_state.TrainState):
    pass

vf_rng = jax.random.PRNGKey(0)
batch_size = 4
dim = 3

# TODO replace with actual independent coupling function once merged
def independent_coupling(
    source: jax.Array,
    target: jax.Array,
) -> tuple[np.ndarray, np.ndarray]:
    src_random_perm_idx = np.random.choice(np.arange(source.shape[0]), size=source.shape[0], replace=False)
    tgt_random_perm_idx = np.random.choice(np.arange(target.shape[0]), size=source.shape[0], replace=False)
    min_shape = min(src_random_perm_idx.shape[0], tgt_random_perm_idx.shape[0])
    return src_random_perm_idx[:min_shape], tgt_random_perm_idx[:min_shape]

def dummy_time_sampler(rng, n):
    return jax.random.uniform(rng, shape=(n,), minval=0.0, maxval=1.0)

class TestJaxMethods:
    @pytest.mark.parametrize("generate_from_noise", [True, False])
    @pytest.mark.parametrize("control_key", [None, "src_cell_data"])
    def test_step_fn_runs_without_error(self, generate_from_noise, control_key):
        vf = MLPUnconditionalVF(
            state_dim=32,
        )
        probability_path = LinearDiracProbabilityPath(sigma=1.0)
        match_fn = independent_coupling
        time_sampler = dummy_time_sampler
        target_dim = 3
        opt = optax.adam(1e-3)
        method = BaseMethod(
            vf=vf,
            probability_path=probability_path,
            time_sampler=time_sampler,
            match_fn=match_fn,
            target_dim=target_dim,
            generate_from_noise=generate_from_noise,
            control_key=control_key,
            optimizer=opt,
            condition_dict={"drug": jnp.ones((batch_size, 1))},
            rng=vf_rng,
        )

        # Create dummy batch
        batch = {
            "src_cell_data": jnp.ones((batch_size, dim)),
            "tgt_cell_data": jnp.ones((batch_size, dim)) * 2,
            "condition": {"dummy_cond": jnp.ones((batch_size, 1))},
        }
        rng = jax.random.PRNGKey(42)
        loss = method.step_fn(rng, batch)

        assert jnp.isscalar(loss) or np.isscalar(loss)
        assert jnp.isfinite(loss)

    @pytest.mark.parametrize("generate_from_noise", [True, False])
    @pytest.mark.parametrize("batched", [True, False])
    def test_predict_batched(self, generate_from_noise, batched):
        state_dim = 3
        vf = MLPUnconditionalVF(
            state_dim=state_dim,
        )
        probability_path = LinearDiracProbabilityPath(sigma=1.0)
        match_fn = independent_coupling
        time_sampler = dummy_time_sampler
        target_dim = dim
        opt = optax.adam(1e-3)
        method = BaseMethod(
            vf=vf,
            probability_path=probability_path,
            time_sampler=time_sampler,
            match_fn=match_fn,
            target_dim=target_dim,
            generate_from_noise=generate_from_noise,
            control_key=None,
            optimizer=opt,
            condition_dict={"drug": jnp.ones((batch_size, 1))},
            rng=vf_rng,
        )
        rng = jax.random.PRNGKey(123)
        src = {
            ("drug_1",): np.random.rand(batch_size, dim),
            ("drug_2",): np.random.rand(batch_size, dim),
        }
        cond = {
            ("drug_1",): {"drug": np.random.rand(1, 1, 3)},
            ("drug_2",): {"drug": np.random.rand(1, 1, 3)},
        }

        x_pred = method.predict(
            x=src, 
            condition=cond, 
            rng=rng,
            batched=batched
        )
        assert x_pred[("drug_1",)].shape == (batch_size, state_dim)
