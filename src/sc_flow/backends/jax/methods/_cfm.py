from typing import Any

import jax
import jax.numpy as jnp

from sc_flow.backends.jax._types import PredictionData
from sc_flow.backends.jax.coupling._coupling import independent_coupling
from sc_flow.backends.jax.methods._methods import JaxGenerativeFlow
from sc_flow.backends.jax.methods._utils import StepData, default_prng_key
from sc_flow.backends.jax.nn._vf import BaseVelocityField, MLPVelocity
from sc_flow.backends.jax.probability_paths._probability_paths import LinearDiracProbabilityPath
from sc_flow.backends.jax.solvers.ode_solver import ODESolver
from sc_flow.backends.jax.solvers.solver import BaseSolver

PyTree = Any


class CFM(JaxGenerativeFlow):
    _module_cls: type[BaseVelocityField] = MLPVelocity
    _default_solver_cls: type[ODESolver] = ODESolver

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self._match_fn is None:
            self._match_fn = independent_coupling
        if self._noise_sampler is None:
            self._noise_sampler = jax.random.normal
        if self._time_sampler is None:
            self._time_sampler = jax.random.uniform
        if self._probability_path is None:
            self._probability_path = LinearDiracProbabilityPath()

    def _prepare_latent_state(
        self,
        rng: jax.Array,
        source: jax.Array | None,
        target_reference: jax.Array,
    ) -> jax.Array:
        if source is None or self._generate_from_noise:
            return self._noise_sampler(rng, target_reference.shape)
        return source

    def _compute_loss(
        self,
        params: PyTree,
        step_data: StepData,
        rng: jax.Array | None = None,
        **kwargs,
    ) -> tuple[jax.Array, dict[str, Any]]:
        target = step_data.target_state
        source = step_data.source_state
        condition_data = self._get_jaxarray_dict_from_data(step_data.target_condition_data)
        group_data = self._get_jaxarray_dict_from_data(step_data.target_group_data)

        rng_noise, rng_time = jax.random.split(default_prng_key(rng))
        latent = self._prepare_latent_state(rng_noise, source, target)

        batch_size = latent.shape[0]
        t = self._time_sampler(rng_time, (batch_size,))
        xt = self._probability_path.compute_xt(t, latent, target)
        ut = self._probability_path.compute_ut(t, xt, latent, target)

        cond = {**condition_data, **group_data} or None
        vt = self._module.apply(
            {"params": params},
            t,
            xt,
            condition_dict=cond,
            source=source,
            train=self._train,
        )
        loss = jnp.mean((vt - ut) ** 2)
        return loss, {"loss": loss}

    def _predict(
        self,
        step_data: StepData,
        *,
        solver_cls: type[BaseSolver] | None = None,
        solver_kwargs: dict[str, Any] | None = None,
        return_trajectory: bool = False,
        num_steps: int = 100,
        latent: jax.Array | None = None,
        **kwargs,
    ) -> PredictionData:
        if latent is None:
            latent = self._prepare_latent_state(default_prng_key(None), step_data.source_state, step_data.target_state)

        condition_reps_dict = self._get_jaxarray_dict_from_data(step_data.target_condition_data)
        group_reps_dict = self._get_jaxarray_dict_from_data(step_data.target_group_data)

        condition_dict = {**condition_reps_dict, **group_reps_dict}

        if solver_kwargs is None:
            solver_kwargs = {}

        if solver_cls is None:
            solver_cls = self._default_solver_cls

        solver = solver_cls(
            self._module,
            method=solver_kwargs.pop("method", "euler"),
            vf_kwargs={"condition_dict": condition_dict, "source": step_data.source_state},
            device_id=self._device,
            **kwargs,
        )
        predictions = solver.solve(
            source=latent,
            t0=0.0,
            t1=1.0,
            num_time_steps=num_steps,
            return_trajectory=return_trajectory,
            solver_kwargs=solver_kwargs,
        )
        if return_trajectory:
            samples = predictions[-1]
            traj = predictions
        else:
            samples = predictions
            traj = None

        return PredictionData(
            samples,
            traj=traj,
        )
