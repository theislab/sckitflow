import functools
from collections.abc import Callable
from typing import Any

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
from flax.training import train_state
from ott.solvers import utils as solver_utils  # TODO: consider implementing this here

import sc_flow.methods as basemethods
from sc_flow import _constants, _types
from sc_flow.backends.jax.methods import _utils

__all__ = ["BaseMethod"] 

LinTerm = tuple[jnp.ndarray, jnp.ndarray]
QuadTerm = tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray | None, jnp.ndarray | None]
GENOTDataMatchFn = Callable[[LinTerm], jnp.ndarray] | Callable[[QuadTerm], jnp.ndarray]


class BaseMethod:
    """TODO."""

    def __init__(
        self,
        vf: Any,  # TODO: adapt type
        probability_path: Any,  # TODO: adapt type
        time_sampler: Callable[[np.ndarray, int], np.ndarray],
        match_fn: Any, # TODO: adapt type
        target_dim: int,
        ema: int = 1,
    ):
        self.vf = vf
        self.probability_path = probability_path
        self.time_sampler = time_sampler
        self.match_fn = match_fn
        self.ema = ema
        # TODO: add cfg

        self._is_trained = False

        self.vf_state = self.vf.create_train_state(
            input_dim=target_dim,
        )
        self.vf_step_fn = self._get_vf_step_fn()

    @staticmethod
    def _prepare_data(
        batch: dict[str, jnp.ndarray],
    ) -> tuple[
        tuple[jnp.ndarray, jnp.ndarray],
        tuple[jnp.ndarray | None, ...],
    ]:
        src_lin, src_quad = batch.get("src_cell_data"), batch.get("src_cell_data_quad")
        tgt_lin, tgt_quad = batch.get("tgt_cell_data"), batch.get("tgt_cell_data_quad")

        if src_quad is None and tgt_quad is None:  # lin
            source, target = src_lin, tgt_lin
            arrs = src_lin, tgt_lin
        elif src_lin is None and tgt_lin is None:  # quad
            source, target = src_quad, tgt_quad
            arrs = src_quad, tgt_quad
        elif all(arr is not None for arr in (src_lin, tgt_lin, src_quad, tgt_quad)):  # fused quad
            source = jnp.concatenate([src_lin, src_quad], axis=1)
            target = jnp.concatenate([tgt_lin, tgt_quad], axis=1)
            arrs = src_quad, tgt_quad, src_lin, tgt_lin
        else:
            raise RuntimeError("Cannot infer OT problem type from data.")

        return (source, target), arrs  # type: ignore[return-value]

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
        rng = jax.random.split(rng, 5)
        rng, rng_resample, rng_noise, rng_time, rng_step_fn = rng

        condition = batch.get("condition")
        (source, target), matching_data = self._prepare_data(batch)
        n = source.shape[0]
        time = self.time_sampler(rng_time, n)
        # latent = self.latent_noise_fn(rng_noise, (n,))

        tmat = self.match_fn(*matching_data)
        src_ixs, tgt_ixs = solver_utils.sample_joint(
            rng_resample,
            tmat,
        )

        source, target = source[src_ixs], target[tgt_ixs]
        loss, self.vf_state = self.vf_step_fn(
            rng_step_fn,
            self.vf_state,
            time,
            source,
            target,
            condition,
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
            return loss, vf_state.apply_gradients(grads=grads)

        return vf_step_fn

    def get_condition_embedding(self, condition: dict[str, jnp.ndarray], return_as_numpy=True) -> jnp.ndarray:
        pass

    def predict(
        self,
        x: jnp.ndarray,
        condition: dict[str, jnp.ndarray] | None = None,
        rng: jnp.ndarray | None = None,
        rng_genot: jnp.ndarray | None = None,
        batched: bool = False,
        **kwargs: Any,
    ) -> jnp.ndarray | tuple[jnp.ndarray, diffrax.Solution]:
        pass

