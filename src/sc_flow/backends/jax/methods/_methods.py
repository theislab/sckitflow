from __future__ import annotations

__all__ = ["JaxBaseMethod", "JaxGenerativeFlow"]

import abc
from typing import Any, Literal, TypeVar

import jax
import jax.numpy as jnp
from flax.training import train_state

from sc_flow.backends.jax._types import PredictionData, TDevice, TMatchFn, TNoiseSamplerFn
from sc_flow.backends.jax._utils import get_jax_device
from sc_flow.backends.jax.methods._opt import JaxOptimizationManager
from sc_flow.backends.jax.methods._utils import StepData
from sc_flow.backends.jax.nn._modules import BaseModule
from sc_flow.backends.jax.probability_paths._probability_paths import BaseProbabilityPath
from sc_flow.backends.jax.solvers.solver import BaseSolver
from sc_flow.data._composite import MatchedDistributions
from sc_flow.data._mixins import BatchMixin
from sc_flow.data.containers._categorical import CategoricalData
from sc_flow.data.containers._coupling import CouplingData
from sc_flow.data.containers._distribution import DistributionData
from sc_flow.data.containers._mixed_type import MixedTypeData
from sc_flow.data.containers._state import StateData
from sc_flow.methods._methods import BaseGenerativeFlow, BaseMethod
from sc_flow.methods._opt import OptimConfig

T = TypeVar("T")
PyTree = Any


class JaxBaseMethod(BaseMethod):
    _module_cls: type[BaseModule] | None = None

    def __init__(
        self,
        *args,
        dtype: jnp.dtype = jnp.float32,
        device_id: TDevice = "cpu",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._dtype = dtype
        self._device = get_jax_device(device_id)
        self._train: bool = False
        self._train_state: train_state.TrainState | None = None
        self._opt_manager: JaxOptimizationManager | None = None

    def init(self, rng: jax.Array, optim_config: OptimConfig) -> None:
        self._opt_manager = JaxOptimizationManager.from_config(optim_config)
        args, kwargs = self._make_init_inputs()
        variables = self._module.init(rng, *args, **kwargs)
        self._train_state = self._opt_manager.create_train_state(
            self._module.apply,
            params=variables["params"],
            batch_stats=variables.get("batch_stats"),
        )

    @abc.abstractmethod
    def _make_init_inputs(self) -> tuple[tuple, dict]:
        """Return ``(args, kwargs)`` for ``module.init``, built from ``dims_registry``."""

    def set_train_mode(self, mode: bool) -> None:
        self._train = mode

    @property
    def params(self) -> PyTree:
        if self._train_state is None:
            raise RuntimeError(f"Call {self.__class__.__name__}.initialize(rng, optim_config) before use.")
        return self._train_state.params

    def extract_state_data(self, state_data: StateData | None) -> jnp.ndarray | None:
        return self._extract_state_data(state_data)

    @staticmethod
    def _safe_subscript_obj(data: T | None, idx: Any | None) -> T | None:
        if data is None:
            return None
        if idx is None:
            return data
        return data[idx]

    def _batchmixin_to_jax(self, batch_mixin: BatchMixin) -> dict[str, jnp.ndarray]:
        return {k: jnp.array(v) for k, v in batch_mixin.mapping.items()}

    def _extract_state_data(self, state_data: StateData | None) -> jnp.ndarray | None:
        if state_data is None:
            return None
        return jnp.array(state_data.X, dtype=self._dtype, device=self._device)

    def _extract_coupling_data(
        self,
        distribution_data: DistributionData,
        mode: Literal["source", "target"],
    ) -> tuple[jnp.ndarray | None, jnp.ndarray | None]:
        coupling_data: CouplingData = getattr(distribution_data, f"{mode}_coupling_data")
        return (
            self._extract_state_data(coupling_data.state_lin),
            self._extract_state_data(coupling_data.state_quad),
        )

    def _extract_distribution_data(
        self,
        distribution_data: DistributionData,
        mode: Literal["source", "target"],
    ) -> tuple[Any, ...]:
        coupling_lin, coupling_quad = self._extract_coupling_data(distribution_data, mode)
        return (
            coupling_lin,
            coupling_quad,
            self._extract_state_data(distribution_data.state_data),
            distribution_data.condition_data,
            distribution_data.groups_data,
        )

    def _get_jaxarray_dict_from_data(
        self,
        data: MixedTypeData | CategoricalData | None,
    ) -> dict[str, jnp.ndarray]:
        if data is None:
            return {}
        return self._batchmixin_to_jax(data.extract_reps())

    def _extract_step_data(self, matched_distr: MatchedDistributions) -> StepData:
        source_data = matched_distr.source_distribution
        target_data = matched_distr.target_distribution

        if target_data is not None:
            target_coupling_lin, target_coupling_quad, target_state, target_cond, target_groups = (
                self._extract_distribution_data(target_data, "target")
            )
        else:
            target_coupling_lin = target_coupling_quad = target_state = target_cond = target_groups = None

        if source_data is not None:
            source_coupling_lin, source_coupling_quad, source_state, source_cond, source_groups = (
                self._extract_distribution_data(source_data, "source")
            )
        else:
            source_coupling_lin = source_coupling_quad = source_state = source_cond = source_groups = None

        return StepData(
            target_state=target_state,
            target_coupling_lin=target_coupling_lin,
            target_coupling_quad=target_coupling_quad,
            target_condition_data=target_cond,
            target_group_data=target_groups,
            source_state=source_state,
            source_coupling_lin=source_coupling_lin,
            source_coupling_quad=source_coupling_quad,
            source_condition_data=source_cond,
            source_group_data=source_groups,
        )


class JaxGenerativeFlow(BaseGenerativeFlow, JaxBaseMethod):
    _default_solver_cls: type[BaseSolver] | None = None

    def __init__(
        self,
        *args,
        probability_path: BaseProbabilityPath | None = None,
        match_fn: TMatchFn | None = None,
        noise_sampler: TNoiseSamplerFn | None = None,
        generate_from_noise: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            probability_path=probability_path,
            match_fn=match_fn,
            noise_sampler=noise_sampler,
            generate_from_noise=generate_from_noise,
            **kwargs,
        )

    def _make_init_inputs(self) -> tuple[tuple, dict]:
        """Build dummy ``(t, x)`` inputs from ``dims_registry`` for ``module.init``.

        Subclasses can override when the module signature differs.
        """
        dummy_t = jnp.zeros((1,))
        dummy_x = jnp.zeros((1, self._dims_registry.state_dim))

        dummy_cond: dict[str, jnp.ndarray] | None = None
        if self._dims_registry.condition_reps_dims:
            dummy_cond = {k: jnp.zeros((1, d)) for k, d in self._dims_registry.condition_reps_dims.items()}

        return (dummy_t, dummy_x), {"condition_dict": dummy_cond}

    @abc.abstractmethod
    def _compute_loss(
        self,
        params: PyTree,
        step_data: StepData,
        rng: jax.Array,
        *args,
        **kwargs,
    ) -> tuple[jnp.ndarray, dict[str, Any]]: ...

    @abc.abstractmethod
    def _predict(
        self,
        step_data: StepData,
        *args,
        **kwargs,
    ) -> PredictionData: ...

    def train_step(
        self,
        matched_distr: MatchedDistributions,
        rng: jax.Array,
        *args,
        **kwargs,
    ) -> dict[str, Any]:
        step_data = self._extract_step_data(matched_distr)
        step_data = self._extract_matched_observations(step_data)

        def loss_fn(params: PyTree) -> tuple[jnp.ndarray, dict[str, Any]]:
            return self._compute_loss(params, step_data, rng, *args, **kwargs)

        self._train_state, metrics = self._opt_manager.step((self._train_state, loss_fn))
        return {k: float(v) for k, v in metrics.items()}

    def predict(
        self,
        matched_distr: MatchedDistributions,
        *args,
        **kwargs,
    ) -> PredictionData:
        step_data = self._extract_step_data(matched_distr)
        return self._predict(step_data, *args, **kwargs)

    def _call_match_fn_safe(
        self,
        source_lin: jnp.ndarray | None,
        source_quad: jnp.ndarray | None,
        target_lin: jnp.ndarray | None,
        target_quad: jnp.ndarray | None,
    ) -> tuple[Any, Any]:
        if source_lin is None and source_quad is None:
            return None, None
        return self._match_fn(
            source_lin=source_lin,
            source_quad=source_quad,
            target_lin=target_lin,
            target_quad=target_quad,
        )

    def _extract_matched_observations(self, step_data: StepData) -> StepData:
        src_idxs, tgt_idxs = self._call_match_fn_safe(
            step_data.source_coupling_lin,
            step_data.source_coupling_quad,
            step_data.target_coupling_lin,
            step_data.target_coupling_quad,
        )

        if src_idxs is None and tgt_idxs is None:
            return step_data

        return StepData(
            target_state=self._safe_subscript_obj(step_data.target_state, tgt_idxs),
            target_coupling_lin=self._safe_subscript_obj(step_data.target_coupling_lin, tgt_idxs),
            target_coupling_quad=self._safe_subscript_obj(step_data.target_coupling_quad, tgt_idxs),
            target_condition_data=self._safe_subscript_obj(step_data.target_condition_data, tgt_idxs),
            target_group_data=self._safe_subscript_obj(step_data.target_group_data, tgt_idxs),
            source_state=self._safe_subscript_obj(step_data.source_state, src_idxs),
            source_coupling_lin=self._safe_subscript_obj(step_data.source_coupling_lin, src_idxs),
            source_coupling_quad=self._safe_subscript_obj(step_data.source_coupling_quad, src_idxs),
            source_condition_data=self._safe_subscript_obj(step_data.source_condition_data, src_idxs),
            source_group_data=self._safe_subscript_obj(step_data.source_group_data, src_idxs),
        )
