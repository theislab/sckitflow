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
from sc_flow.backends.jax._types import ArrayLike
from sc_flow.backends.jax.methods import _utils

__all__ = ["BaseMethod"] 

LinTerm = tuple[jnp.ndarray, jnp.ndarray] # Arraylike instead of jnp.ndarray?
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
        generate_from_noise: bool = False,
        control_key: str | None = None,
        noise_distribution: Callable[[jax.Array, tuple[int, ...]], jnp.ndarray] | None = None,
        **kwargs: Any,
    ):
        self.vf = vf
        self.probability_path = probability_path
        self.time_sampler = time_sampler
        self.match_fn = match_fn
        self.ema = ema
        self.generate_from_noise = generate_from_noise
        self.control_key = control_key
        self.noise_distribution = noise_distribution or (
            lambda rng, shape: jax.random.normal(rng, shape)
        )
        # TODO: add cfg

        self._is_trained = False
        self.vf_state = self.vf.create_train_state(
            input_dim=target_dim,
            **kwargs
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

        src_ixs, tgt_ixs = self.match_fn(*matching_data)

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
                rng_flow, rng_latent, rng_dropout = jax.random.split(rng, 3)
                if self.generate_from_noise:
                    latent = self.noise_distribution(rng_latent, target.shape)
                    x_t = self.probability_path.compute_xt(t, latent, target, rng_flow)
                    if self.control_key is not None:
                        conditions = conditions.copy()
                        conditions[self.control_key] = source
                else:
                    x_t = self.probability_path.compute_xt(t, source, target, rng_flow)
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
        cond_embedding = self.vf.apply(
            {"params": self.vf_state.params},
            condition,
            method="get_condition_embedding",
        )
        return np.asarray(cond_embedding) if return_as_numpy else cond_embedding

    def predict(
        self,
        x: ArrayLike,
        condition: dict[str, ArrayLike] | None = None,
        rng: ArrayLike | None = None,
        batched: bool = False,
        **kwargs: Any,
    ) -> ArrayLike | tuple[ArrayLike, diffrax.Solution]:
        ''' TODO '''
        if batched and not x:
            return {}

        if batched:
            keys = sorted(x.keys())
            condition_keys = sorted(set().union(*(condition[k].keys() for k in keys)))
            _predict_jit = jax.jit(lambda x, condition: self._predict_jit(x, condition, rng, **kwargs))
            batched_predict = jax.vmap(_predict_jit, in_axes=(0, dict.fromkeys(condition_keys, 0)))
            # assert that the number of cells is the same for each condition
            n_cells = x[keys[0]].shape[0]
            for k in keys:
                assert x[k].shape[0] == n_cells, "The number of cells must be the same for each condition"
            src_inputs = jnp.stack([x[k] for k in keys], axis=0)
            batched_conditions = {}
            for cond_key in condition_keys:
                batched_conditions[cond_key] = jnp.stack([condition[k][cond_key] for k in keys])

            pred_targets = batched_predict(src_inputs, batched_conditions)
            return {k: pred_targets[i] for i, k in enumerate(keys)}
        elif isinstance(x, dict):
            return jax.tree.map(
                functools.partial(self._predict_jit, rng=rng, **kwargs),
                x,
                condition,
            )
        else:
            x_pred = self._predict_jit(x, condition, rng, **kwargs)
            return np.array(x_pred)

    def _predict_jit(
        self,
        x: jnp.ndarray,
        condition: dict[str, jnp.ndarray] | None = None,
        rng: jnp.ndarray | None = None,
        **kwargs: Any,
    ) -> jnp.ndarray | tuple[jnp.ndarray, diffrax.Solution]:
        ''' TODO '''
        kwargs.setdefault("dt0", None)
        kwargs.setdefault("solver", diffrax.Tsit5())
        kwargs.setdefault("stepsize_controller", diffrax.PIDController(rtol=1e-5, atol=1e-5))
        rng = jax.random.key(0) if rng is None else rng

        if self.generate_from_noise:
            latent = self.noise_distribution(rng, x.shape)
            def vf(t: float, x: jnp.ndarray, args: tuple[dict[str, jnp.ndarray], jnp.ndarray]) -> jnp.ndarray:
                params = self.vf_state.params
                source, condition = args
                return self.vf_state.apply_fn({"params": params}, t, x, condition, source)

            def solve_ode(latent: jnp.ndarray, x: jnp.ndarray, condition: dict[str, jnp.ndarray]) -> jnp.ndarray:
                term = diffrax.ODETerm(vf)
                sol = diffrax.diffeqsolve(
                    term,
                    t0=0.0,
                    t1=1.0,
                    y0=latent,
                    args=(x, condition),
                    **kwargs,
                )
                return sol.ys[0]
            x_pred = solve_ode(latent, x, condition)
            # x_pred = jax.jit(jax.vmap(solve_ode, in_axes=[0, 0, None]))(latent, x, condition)
        else:
            def vf(t: ArrayLike, x: ArrayLike, args: tuple[dict[str, ArrayLike], ArrayLike]) -> jnp.ndarray:
                params = self.vf_state.params
                condition = args
                return self.vf_state.apply_fn({"params": params}, t, x, condition)

            def solve_ode(x: jnp.ndarray, condition: dict[str, jnp.ndarray]) -> jnp.ndarray:
                term = diffrax.ODETerm(vf)
                sol = diffrax.diffeqsolve(
                    term,
                    t0=0.0,
                    t1=1.0,
                    y0=x,
                    args=condition,
                    **kwargs,
                ) 
                return sol.ys[0]

            # x_pred = jax.jit(jax.vmap(solve_ode, in_axes=[0, None]))(x, condition)
            x_pred = solve_ode(x, condition)
        return x_pred
    
    def validation_step( # pass rng?
        self,
        batch: dict[str, ArrayLike], 
    ) -> ArrayLike:
        """Validation step of the trainer.

        Parameters
        ----------
        batch
            Data batch with keys ``src_cell_data``, ``tgt_cell_data``, and
            ``condition``.
        Returns
        -------
            prediction: The dictionary of matched predictions and targets to compute validation on
        """
        (source, _), _ = self._prepare_data(batch)
        condition = batch.get("condition")

        x_pred = self.predict(
            x=source,
            condition=condition,
            batched=False,
        )
        return x_pred
    
    def train_step(
        self,
        batch: dict[str, ArrayLike],
        rng_step_fn: ArrayLike | None = None,
    ) -> float:
        """Training step of the trainer.

        Parameters
        ----------
        batch
            Data batch with keys ``src_cell_data``, ``tgt_cell_data``, and
            ``condition``.
        rng_step_fn
            Random number generator for the step function.
        Returns
        -------
            float: The value of the loss computed on a batch
        """
        rng = jax.random.key(0) if rng_step_fn is None else rng_step_fn
        loss = self.step_fn(rng, batch)
        return float(loss)