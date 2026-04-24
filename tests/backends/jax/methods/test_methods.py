from __future__ import annotations

import functools
from unittest.mock import Mock

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import linen as nn
from flax.training import train_state

from sc_flow.backends.jax._types import PredictionData
from sc_flow.backends.jax.methods._methods import JaxBaseMethod, JaxGenerativeFlow
from sc_flow.backends.jax.methods._opt import JaxOptimizationManager, TrainStateWithBatchStats
from sc_flow.backends.jax.methods._utils import StepData
from sc_flow.backends.jax.nn._modules import BaseModule
from sc_flow.data._composite import MatchedDistributions
from sc_flow.data.containers._coupling import CouplingData
from sc_flow.data.containers._distribution import DistributionData
from sc_flow.data.containers._state import StateData
from sc_flow.methods._opt import OptimConfig


# ---------------------------------------------------------------------------
# Minimal Flax module — no BatchNorm, variables has only "params"
# ---------------------------------------------------------------------------
class DummyFlaxModule(BaseModule):
    state_dim: int = 2

    @nn.compact
    def __call__(self, t, x, condition_dict=None, train: bool = False):
        return nn.Dense(self.state_dim)(x)

    @classmethod
    def init_from_dims_registry(cls, dims_registry, **kwargs):
        return cls(state_dim=dims_registry.state_dim)


# ---------------------------------------------------------------------------
# Concrete subclasses for testing
# ---------------------------------------------------------------------------
class DummyJaxMethod(JaxBaseMethod):
    _module_cls = DummyFlaxModule

    def _make_init_inputs(self):
        dummy_t = jnp.zeros((1,))
        dummy_x = jnp.zeros((1, self._dims_registry.state_dim))
        return (dummy_t, dummy_x), {}

    def train_step(self, *args, **kwargs):
        return {}

    def predict(self, *args, **kwargs):
        return None


class DummyJaxGenerativeFlow(JaxGenerativeFlow):
    _module_cls = DummyFlaxModule
    _default_solver_cls = Mock()

    def _compute_loss(self, params, step_data, rng, **kwargs):
        out = self._module.apply({"params": params}, jnp.zeros((1,)), step_data.target_state)
        loss = jnp.mean(out**2)
        return loss, {"loss": loss}

    def _predict(self, step_data, *args, **kwargs):
        return PredictionData(samples=jnp.zeros((1, 2)), traj=None)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def rng():
    return jax.random.key(0)


@pytest.fixture
def mock_dims_registry():
    reg = Mock()
    reg.state_dim = 2
    reg.condition_reps_dims = None
    reg.feature_names = ["g1", "g2"]
    reg.n_features = 2
    return reg


@pytest.fixture
def mock_data_manager():
    return Mock()


@pytest.fixture
def optim_config():
    return OptimConfig(lr=1e-3)


@pytest.fixture
def jax_method(mock_dims_registry, mock_data_manager):
    return DummyJaxMethod(mock_dims_registry, mock_data_manager, False)


@pytest.fixture
def jax_gen_flow(mock_dims_registry, mock_data_manager):
    flow = DummyJaxGenerativeFlow(mock_dims_registry, mock_data_manager, False)
    flow._match_fn = Mock(return_value=(None, None))
    return flow


@pytest.fixture
def initialized_jax_method(jax_method, rng, optim_config):
    jax_method.init(rng, optim_config)
    return jax_method


@pytest.fixture
def initialized_gen_flow(jax_gen_flow, rng, optim_config):
    jax_gen_flow.init(rng, optim_config)
    return jax_gen_flow


def _make_step_data(batch_size: int = 4, state_dim: int = 2) -> StepData:
    x = jnp.ones((batch_size, state_dim))
    return StepData(
        target_state=x,
        target_coupling_lin=x,
        target_coupling_quad=None,
        target_condition_data=None,
        target_group_data=None,
        source_state=x,
        source_coupling_lin=x,
        source_coupling_quad=None,
        source_condition_data=None,
        source_group_data=None,
    )


def _make_matched_distributions(batch_size: int = 4, state_dim: int = 2) -> MatchedDistributions:
    X = np.ones((batch_size, state_dim), dtype=np.float32)
    state = StateData(X=X)
    coupling = Mock(spec=CouplingData)
    coupling.state_lin = state
    coupling.state_quad = None

    dist = Mock(spec=DistributionData)
    dist.state_data = state
    dist.condition_data = None
    dist.groups_data = None
    dist.source_coupling_data = coupling
    dist.target_coupling_data = coupling

    matched = Mock(spec=MatchedDistributions)
    matched.source_distribution = dist
    matched.target_distribution = dist
    return matched


# ===========================================================================
# JaxOptimizationManager
# ===========================================================================
class TestJaxOptimizationManager:
    def test_from_config_default_adam(self, optim_config):
        manager = JaxOptimizationManager.from_config(optim_config)
        assert isinstance(manager, JaxOptimizationManager)

    def test_from_config_string_optimizer(self):
        config = OptimConfig(optimizer_cls="sgd", lr=1e-2)
        manager = JaxOptimizationManager.from_config(config)
        assert manager is not None

    def test_from_config_callable_optimizer(self):
        config = OptimConfig(optimizer_cls=optax.rmsprop, lr=1e-3)
        manager = JaxOptimizationManager.from_config(config)
        assert manager is not None

    def test_from_config_pre_created_optimizer(self):
        tx = optax.adam(1e-3)
        config = OptimConfig(optimizer=tx)
        manager = JaxOptimizationManager.from_config(config)
        assert manager.tx is tx

    def test_from_config_invalid_optimizer_string(self):
        config = OptimConfig(optimizer_cls="not_a_real_optimizer")
        with pytest.raises(ValueError, match="not found in optax"):
            JaxOptimizationManager.from_config(config)

    def test_from_config_invalid_optimizer_type(self):
        config = OptimConfig(optimizer_cls=42)
        with pytest.raises(TypeError):
            JaxOptimizationManager.from_config(config)

    def test_from_config_with_schedule_instance(self):
        schedule = optax.cosine_decay_schedule(init_value=1e-3, decay_steps=100)
        config = OptimConfig(lr_scheduler=schedule)
        manager = JaxOptimizationManager.from_config(config)
        assert manager is not None

    def test_from_config_with_schedule_string(self):
        config = OptimConfig(
            lr_scheduler_cls="cosine_decay_schedule",
            lr_scheduler_kwargs={"init_value": 1e-3, "decay_steps": 100},
        )
        manager = JaxOptimizationManager.from_config(config)
        assert manager is not None

    def test_from_config_invalid_schedule_string(self):
        config = OptimConfig(lr_scheduler_cls="not_a_real_schedule")
        with pytest.raises(ValueError, match="not found in optax"):
            JaxOptimizationManager.from_config(config)

    def test_from_config_invalid_schedule_type(self):
        config = OptimConfig(lr_scheduler_cls=42)
        with pytest.raises(TypeError):
            JaxOptimizationManager.from_config(config)

    def test_lr_scheduler_step_attribute(self):
        config = OptimConfig(lr=1e-3, lr_scheduler_step="epoch")
        manager = JaxOptimizationManager.from_config(config)
        assert manager.lr_scheduler_step == "epoch"

    def test_create_train_state_no_batch_stats(self, optim_config):
        manager = JaxOptimizationManager.from_config(optim_config)
        params = {"Dense_0": {"kernel": jnp.ones((2, 2)), "bias": jnp.zeros((2,))}}
        ts = manager.create_train_state(apply_fn=None, params=params)
        assert isinstance(ts, train_state.TrainState)
        assert not isinstance(ts, TrainStateWithBatchStats)

    def test_create_train_state_with_batch_stats(self, optim_config):
        manager = JaxOptimizationManager.from_config(optim_config)
        params = {"Dense_0": {"kernel": jnp.ones((2, 2))}}
        batch_stats = {"BatchNorm_0": {"mean": jnp.zeros((2,)), "var": jnp.ones((2,))}}
        ts = manager.create_train_state(apply_fn=None, params=params, batch_stats=batch_stats)
        assert isinstance(ts, TrainStateWithBatchStats)
        assert ts.batch_stats is batch_stats

    def test_step_updates_params(self, optim_config):
        manager = JaxOptimizationManager.from_config(optim_config)
        params = {"w": jnp.ones((2,))}
        ts = manager.create_train_state(apply_fn=None, params=params)
        params_before = ts.params["w"]

        def loss_fn(p):
            loss = jnp.sum(p["w"] ** 2)
            return loss, {"loss": loss}

        new_ts, metrics = manager.step((ts, loss_fn))
        assert not jnp.allclose(new_ts.params["w"], params_before)
        assert "loss" in metrics

    def test_step_increments_step_counter(self, optim_config):
        manager = JaxOptimizationManager.from_config(optim_config)
        params = {"w": jnp.ones((2,))}
        ts = manager.create_train_state(apply_fn=None, params=params)
        assert ts.step == 0

        def loss_fn(p):
            return jnp.sum(p["w"] ** 2), {"loss": jnp.array(0.0)}

        new_ts, _ = manager.step((ts, loss_fn))
        assert new_ts.step == 1

    def test_step_returns_metrics_without_batch_stats_key(self, optim_config):
        manager = JaxOptimizationManager.from_config(optim_config)
        params = {"w": jnp.ones((2,))}
        ts = manager.create_train_state(apply_fn=None, params=params)

        def loss_fn(p):
            return jnp.sum(p["w"]), {"mse": jnp.array(1.0), "kl": jnp.array(0.5)}

        _, metrics = manager.step((ts, loss_fn))
        assert set(metrics.keys()) == {"mse", "kl"}
        assert "batch_stats" not in metrics

    def test_step_updates_batch_stats_when_present(self, optim_config):
        manager = JaxOptimizationManager.from_config(optim_config)
        params = {"w": jnp.ones((2,))}
        initial_stats = {"mean": jnp.zeros((2,))}
        ts = manager.create_train_state(apply_fn=None, params=params, batch_stats=initial_stats)
        new_stats = {"mean": jnp.ones((2,))}

        def loss_fn(p):
            return jnp.sum(p["w"]), {"loss": jnp.array(1.0), "batch_stats": new_stats}

        new_ts, metrics = manager.step((ts, loss_fn))
        assert isinstance(new_ts, TrainStateWithBatchStats)
        assert jnp.allclose(new_ts.batch_stats["mean"], jnp.ones((2,)))
        assert "batch_stats" not in metrics


# ===========================================================================
# JaxBaseMethod
# ===========================================================================
class TestJaxBaseMethod:
    def test_init_state_before_init(self, jax_method):
        assert jax_method._train_state is None
        assert jax_method._opt_manager is None

    def test_params_raises_before_init(self, jax_method):
        with pytest.raises(RuntimeError):
            _ = jax_method.params

    def test_init_creates_train_state(self, initialized_jax_method):
        assert initialized_jax_method._train_state is not None
        assert isinstance(initialized_jax_method._train_state, train_state.TrainState)

    def test_init_creates_opt_manager(self, initialized_jax_method):
        assert initialized_jax_method._opt_manager is not None
        assert isinstance(initialized_jax_method._opt_manager, JaxOptimizationManager)

    def test_params_accessible_after_init(self, initialized_jax_method):
        params = initialized_jax_method.params
        assert params is not None
        assert isinstance(params, dict)

    def test_set_train_mode(self, initialized_jax_method):
        initialized_jax_method.set_train_mode(True)
        assert initialized_jax_method._train is True
        initialized_jax_method.set_train_mode(False)
        assert initialized_jax_method._train is False

    def test_extract_state_data_none(self, initialized_jax_method):
        assert initialized_jax_method.extract_state_data(None) is None

    def test_extract_state_data_converts_to_jax(self, initialized_jax_method):
        state = StateData(X=np.random.randn(5, 2).astype(np.float32))
        result = initialized_jax_method.extract_state_data(state)
        assert isinstance(result, jnp.ndarray)
        assert result.shape == (5, 2)

    def test_safe_subscript_obj_none_data(self, jax_method):
        assert jax_method._safe_subscript_obj(None, jnp.array([0, 1])) is None

    def test_safe_subscript_obj_none_idx(self, jax_method):
        data = jnp.ones((4, 2))
        assert jax_method._safe_subscript_obj(data, None) is data

    def test_safe_subscript_obj_indexes(self, jax_method):
        data = jnp.arange(8).reshape(4, 2)
        idx = jnp.array([0, 2])
        result = jax_method._safe_subscript_obj(data, idx)
        assert result.shape == (2, 2)
        assert jnp.allclose(result[0], jnp.array([0, 1]))
        assert jnp.allclose(result[1], jnp.array([4, 5]))

    def test_batchmixin_to_jax(self, jax_method):
        batch_mixin = Mock()
        batch_mixin.mapping = {"a": np.array([1.0, 2.0]), "b": np.array([3.0, 4.0])}
        result = jax_method._batchmixin_to_jax(batch_mixin)
        assert isinstance(result["a"], jnp.ndarray)
        assert result["a"].tolist() == [1.0, 2.0]

    def test_extract_step_data_both_distributions(self, jax_method):
        matched = _make_matched_distributions()
        step_data = jax_method._extract_step_data(matched)
        assert isinstance(step_data, StepData)
        assert step_data.target_state is not None
        assert step_data.source_state is not None

    def test_extract_step_data_no_source(self, jax_method):
        matched = Mock(spec=MatchedDistributions)
        matched.source_distribution = None
        state = StateData(X=np.ones((4, 2), dtype=np.float32))
        coupling = Mock(spec=CouplingData)
        coupling.state_lin = state
        coupling.state_quad = None
        dist = Mock(spec=DistributionData)
        dist.state_data = state
        dist.condition_data = None
        dist.groups_data = None
        dist.target_coupling_data = coupling
        matched.target_distribution = dist

        step_data = jax_method._extract_step_data(matched)
        assert step_data.source_state is None
        assert step_data.source_coupling_lin is None


# ===========================================================================
# JaxGenerativeFlow
# ===========================================================================
class TestJaxGenerativeFlow:
    def test_make_init_inputs_no_condition(self, jax_gen_flow):
        args, kwargs = jax_gen_flow._make_init_inputs()
        t, x = args
        assert t.shape == (1,)
        assert x.shape == (1, jax_gen_flow._dims_registry.state_dim)
        assert kwargs["condition_dict"] is None

    def test_make_init_inputs_with_condition(self, mock_dims_registry, mock_data_manager):
        mock_dims_registry.condition_reps_dims = {"pert": 4}
        flow = DummyJaxGenerativeFlow(mock_dims_registry, mock_data_manager, False)
        _, kwargs = flow._make_init_inputs()
        assert kwargs["condition_dict"] is not None
        assert "pert" in kwargs["condition_dict"]
        assert kwargs["condition_dict"]["pert"].shape == (1, 4)

    def test_call_match_fn_safe_no_source(self, jax_gen_flow):
        src_idx, tgt_idx = jax_gen_flow._call_match_fn_safe(None, None, jnp.ones((4, 2)), None)
        assert src_idx is None
        assert tgt_idx is None
        jax_gen_flow._match_fn.assert_not_called()

    def test_call_match_fn_safe_with_source(self, jax_gen_flow):
        src_lin = jnp.ones((4, 2))
        tgt_lin = jnp.ones((4, 2))
        expected = (jnp.array([0, 1]), jnp.array([2, 3]))
        jax_gen_flow._match_fn.return_value = expected
        src_idx, tgt_idx = jax_gen_flow._call_match_fn_safe(src_lin, None, tgt_lin, None)
        jax_gen_flow._match_fn.assert_called_once_with(
            source_lin=src_lin, source_quad=None, target_lin=tgt_lin, target_quad=None
        )
        assert jnp.array_equal(src_idx, expected[0])
        assert jnp.array_equal(tgt_idx, expected[1])

    def test_extract_matched_observations_no_match(self, jax_gen_flow):
        step_data = _make_step_data()
        jax_gen_flow._match_fn = Mock(return_value=(None, None))
        result = jax_gen_flow._extract_matched_observations(step_data)
        assert result.target_state is step_data.target_state
        assert result.source_state is step_data.source_state

    def test_extract_matched_observations_with_match(self, jax_gen_flow):
        step_data = _make_step_data(batch_size=4)
        src_idx = jnp.array([0, 2])
        tgt_idx = jnp.array([1, 3])
        jax_gen_flow._match_fn = Mock(return_value=(src_idx, tgt_idx))
        result = jax_gen_flow._extract_matched_observations(step_data)
        assert result.target_state.shape[0] == 2
        assert result.source_state.shape[0] == 2

    def test_train_step_returns_float_metrics(self, initialized_gen_flow, rng):
        matched = _make_matched_distributions()
        metrics = initialized_gen_flow.train_step(matched, rng)
        assert isinstance(metrics, dict)
        assert "loss" in metrics
        assert isinstance(metrics["loss"], float)

    def test_train_step_increments_step(self, initialized_gen_flow, rng):
        matched = _make_matched_distributions()
        initialized_gen_flow.train_step(matched, rng)
        initialized_gen_flow.train_step(matched, rng)
        assert initialized_gen_flow._train_state.step == 2

    def test_train_step_updates_params(self, initialized_gen_flow, rng):
        matched = _make_matched_distributions()
        params_before = jax.tree.map(jnp.copy, initialized_gen_flow.params)
        initialized_gen_flow.train_step(matched, rng)
        # Step counter advancing confirms opt manager ran
        assert initialized_gen_flow._train_state.step == 1
        # At least one leaf should differ
        leaves_before = jax.tree_util.tree_leaves(params_before)
        leaves_after = jax.tree_util.tree_leaves(initialized_gen_flow.params)
        assert any(not jnp.allclose(b, a) for b, a in zip(leaves_before, leaves_after, strict=False))

    def test_predict_returns_prediction_data(self, initialized_gen_flow):
        matched = _make_matched_distributions()
        result = initialized_gen_flow.predict(matched)
        assert isinstance(result, PredictionData)
        assert result.samples is not None

    def test_loss_fn_via_functools_partial(self, initialized_gen_flow, rng):
        step_data = _make_step_data()
        loss_fn = functools.partial(initialized_gen_flow._compute_loss, step_data=step_data, rng=rng)
        loss, aux = loss_fn(initialized_gen_flow.params)
        assert jnp.isfinite(loss)
        assert "loss" in aux
