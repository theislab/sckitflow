from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax.training import train_state
from ott.solvers import utils as solver_utils  # TODO: consider implementing this here

import sc_flow.methods as basemethods
from sc_flow.backends.methods import _utils

__all__ = ["FlowMatching", "OTFlowMatching", "GENOT"]


class FlowMatching(basemethods.FlowMatching):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class OTFlowMatching(basemethods.OTFlowMatching):
    def __init__(self, match_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] | None = None, **kwargs):
        super().__init__(**kwargs)
        self.match_fn = match_fn

    def step_fn(
        self,
        rng: jnp.ndarray,
        batch: dict[str, jnp.ndarray],
    ) -> float:
        """Single step function of the solver.

        Parameters
        ----------
        rng
            Random number generator.
        batch
            Data batch with keys ``src_cell_data``, ``tgt_cell_data``, and
            ``condition``.

        Returns
        -------
        Loss value.
        """
        src, tgt = batch["src_cell_data"], batch["tgt_cell_data"]
        condition = batch.get("condition")
        rng_resample, rng_time, rng_step_fn = jax.random.split(rng, 3)
        n = src.shape[0]
        time = self.time_sampler(rng_time, n)

        if self.match_fn is not None:
            tmat = self.match_fn(src, tgt)
            src_ixs, tgt_ixs = solver_utils.sample_joint(rng_resample, tmat)
            src, tgt = src[src_ixs], tgt[tgt_ixs]

        self.vf_state, loss = self.vf_step_fn(
            rng_step_fn,
            self.vf_state,
            time,
            src,
            tgt,
            condition,
        )

        if self.ema == 1.0:
            self.vf_state_inference = self.vf_state
        else:
            self.vf_state_inference = self.vf_state_inference.replace(
                params=_utils.ema_update(self.vf_state_inference.params, self.vf_state.params, self.ema)
            )
        return loss

    def _get_vf_step_fn(self) -> Callable:  # type: ignore[type-arg]
        @jax.jit
        def vf_step_fn(
            rng: jax.Array,
            vf_state: train_state.TrainState,
            time: jnp.ndarray,
            source: jnp.ndarray,
            target: jnp.ndarray,
            conditions: dict[str, jnp.ndarray],
        ):
            def loss_fn(
                params: jnp.ndarray,
                t: jnp.ndarray,
                source: jnp.ndarray,
                target: jnp.ndarray,
                conditions: dict[str, jnp.ndarray],
                rng: jax.Array,
            ) -> jnp.ndarray:
                rng_flow, rng_dropout = jax.random.split(rng, 2)
                x_t = self.probability_path.compute_xt(rng_flow, t, source, target)
                v_t = vf_state.apply_fn(
                    {"params": params},
                    t,
                    x_t,
                    conditions,
                    rngs={"dropout": rng_dropout},
                )
                u_t = self.probability_path.compute_ut(t, x_t, source, target)
                return jnp.mean((v_t - u_t) ** 2)

            grad_fn = jax.value_and_grad(loss_fn)
            loss, grads = grad_fn(vf_state.params, time, source, target, conditions, rng)
            return vf_state.apply_gradients(grads=grads), loss

        return vf_step_fn


class GENOT(basemethods.GENOT):
    def __init__(self, match_fn: Callable, **kwargs):
        super().__init__(**kwargs)
        self.match_fn = match_fn
