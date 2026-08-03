from unittest.mock import Mock

import lightning.pytorch as pl
import pytest
import torch

from sckitflow.core._types import PredictionData, StepData
from sckitflow.core.methods._base import BaseMethod, GenerativeFlow
from sckitflow.core.methods._opt import OptimConfig
from sckitflow.core.nn._modules import BaseModule
from sckitflow.data._dims_registry import DataDimensionalitiesRegistry
from sckitflow.data._manager import DataManager


# -----------------------------------------------------------------------------
# Minimal torch module that can be instantiated
# -----------------------------------------------------------------------------
class DummyModule(BaseModule):
    """A minimal module with parameters to satisfy optimizers."""

    def __init__(self):
        super().__init__()
        # Real modules build their layers in `__init__` via `_make_modules`; without this
        # the module has no parameters and `configure_optimizers` has nothing to optimize.
        self.linear = self._make_modules()

    def _make_modules(self):
        return torch.nn.Linear(2, 2)

    def forward(self, t, x, condition_dict=None, source=None):
        return self.linear(x)


# -----------------------------------------------------------------------------
# Concrete test subclasses that use DummyModule
# -----------------------------------------------------------------------------
class DummyMethod(BaseMethod):
    _module_cls = DummyModule

    def compute_loss(self, *args, **kwargs):
        return torch.tensor(0.0), {}

    def infer(self, *args, **kwargs):
        pass


class DummyGenerativeFlow(GenerativeFlow):
    _module_cls = DummyModule
    _default_solver_cls = Mock()

    def compute_loss(self, step_data, *args, **kwargs):
        return torch.tensor(0.0), {}

    def infer(self, step_data, *args, **kwargs):
        return PredictionData(X=torch.randn(1, 2), traj=None)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture
def mock_dims_registry():
    reg = Mock(spec=DataDimensionalitiesRegistry)
    reg.feature_names = ["g1", "g2"]
    reg.n_features = 2
    return reg


@pytest.fixture
def mock_data_manager():
    """An unpaired data manager: no control values and no matched keys."""
    dm = Mock(spec=DataManager)
    dm.control_values_dict = None
    dm.matched_keys = None
    return dm


@pytest.fixture
def mock_paired_data_manager():
    dm = Mock(spec=DataManager)
    dm.control_values_dict = {"drug": "control"}
    dm.matched_keys = None
    return dm


@pytest.fixture
def method(mock_dims_registry, mock_data_manager):
    return DummyMethod(
        dims_registry=mock_dims_registry,
        dm=mock_data_manager,
        dtype=torch.float32,
        device_id="cpu",
    )


@pytest.fixture
def gen_flow(mock_dims_registry, mock_paired_data_manager):
    flow = DummyGenerativeFlow(
        dims_registry=mock_dims_registry,
        dm=mock_paired_data_manager,
        dtype=torch.float32,
        device_id="cpu",
    )
    flow._match_fn = Mock(return_value=(None, None))
    return flow


# -----------------------------------------------------------------------------
# Test suite for BaseMethod
# -----------------------------------------------------------------------------
class TestBaseMethod:
    """Tests for the abstract BaseMethod class."""

    def test_abstract_class_cannot_be_instantiated(self):
        """BaseMethod should raise TypeError if abstract methods not implemented."""
        with pytest.raises(TypeError):
            BaseMethod(dims_registry=Mock(), dm=Mock())

    def test_missing_module_cls_raises(self, mock_dims_registry, mock_data_manager):
        """A subclass that forgets `_module_cls` fails loudly at construction."""

        class NoModule(BaseMethod):
            def compute_loss(self, *args, **kwargs):
                return torch.tensor(0.0), {}

            def infer(self, *args, **kwargs):
                pass

        with pytest.raises(NotImplementedError, match="_module_cls"):
            NoModule(mock_dims_registry, mock_data_manager)

    def test_is_a_lightning_module(self, method):
        """The method is the LightningModule the trainer fits."""
        assert isinstance(method, pl.LightningModule)

    def test_concrete_method_instantiation(self, mock_dims_registry, mock_data_manager):
        """Concrete subclass should be instantiable and set attributes correctly."""
        method = DummyMethod(mock_dims_registry, mock_data_manager)
        assert method._dims_registry is mock_dims_registry
        assert method._dm is mock_data_manager
        assert method.is_paired_setting is False  # derived from the data manager
        assert method._module is not None  # from _module_cls.init_from_dims_registry

    def test_is_paired_setting_derived_from_dm(self, mock_dims_registry, mock_paired_data_manager):
        """is_paired_setting is derived from the data manager's control/matched config."""
        method = DummyMethod(mock_dims_registry, mock_paired_data_manager)
        assert method.is_paired_setting is True

    def test_properties_return_correct_values(self, mock_dims_registry, mock_paired_data_manager):
        method = DummyMethod(mock_dims_registry, mock_paired_data_manager)
        assert method.module is not None
        assert method.dm is mock_paired_data_manager
        assert method.dims_registry is mock_dims_registry
        assert method.is_paired_setting is True

    def test_module_is_registered_as_a_child(self, method):
        """Lightning has to see the module to place it on a device and save its weights."""
        assert method.module in set(method.modules())
        assert any(p is not None for p in method.parameters())

    def test_device_and_dtype_track_to(self, method):
        """`device`/`dtype` come from Lightning, so moving the method moves the batches."""
        assert method.device.type == "cpu"
        assert method.dtype is torch.float32
        method.to(torch.float64)
        assert method.dtype is torch.float64

    def test_safe_subscript_obj(self, method):
        data = torch.tensor([1, 2, 3, 4])
        idx = torch.tensor([0, 2])
        result = method._safe_subscript_obj(data, idx)
        assert torch.equal(result, torch.tensor([1, 3]))
        assert method._safe_subscript_obj(None, idx) is None
        assert method._safe_subscript_obj(data, None) is data

    def test_transfer_batch_to_device_is_a_passthrough(self, method):
        """Nodes are frozen dataclasses of numpy arrays; Lightning must not walk them."""
        node = Mock()
        assert method.transfer_batch_to_device(node, torch.device("cpu"), 0) is node

    def test_training_step_returns_loss_and_logs_metrics(self, method):
        """Lightning takes the returned loss and runs backward/step itself."""
        node = Mock()
        node.n_target_obs = 4
        method.train_step = Mock(return_value=(torch.tensor(0.25), {"loss": 0.25}))
        method.log_dict = Mock()

        loss = method.training_step(node, 0)

        assert loss.item() == pytest.approx(0.25)
        method.train_step.assert_called_once_with(node)
        logged = method.log_dict.call_args
        assert logged.args[0] == {"loss": 0.25}
        assert logged.kwargs["batch_size"] == 4
        # One epoch only, so per-step is the only meaningful granularity.
        assert logged.kwargs["on_step"] is True
        assert logged.kwargs["on_epoch"] is False

    def test_validation_step_returns_predictions_and_targets(self, method):
        """Raw arrays are handed back so metric callbacks can accumulate across nodes."""
        node = Mock()
        node.target_distr.state_data.X = torch.randn(4, 2)
        preds = PredictionData(X=torch.randn(4, 2), traj=None)
        method.predict = Mock(return_value=preds)
        method.val_predict_kwargs = {"n_samples": 3}

        out = method.validation_step(node, 0)

        method.predict.assert_called_once_with(node, n_samples=3)
        assert out["predictions"] is preds.X
        assert out["targets"] is node.target_distr.state_data.X

    def test_validation_step_tolerates_missing_target_state(self, method):
        """Predicting from metadata alone leaves no target to compare against."""
        node = Mock()
        node.target_distr.state_data = None
        method.predict = Mock(return_value=PredictionData(X=torch.randn(4, 2), traj=None))

        assert method.validation_step(node, 0)["targets"] is None

    def test_configure_optimizers_uses_optim_config(self, mock_dims_registry, mock_data_manager):
        config = OptimConfig(optimizer_cls="SGD", lr=0.1)
        method = DummyMethod(mock_dims_registry, mock_data_manager, optim_config=config)

        resolved = method.configure_optimizers()

        assert isinstance(resolved["optimizer"], torch.optim.SGD)
        assert resolved["optimizer"].param_groups[0]["lr"] == pytest.approx(0.1)
        assert "lr_scheduler" not in resolved

    def test_default_optim_config_is_used_when_unset(self, method):
        assert isinstance(method.optim_config, OptimConfig)
        assert isinstance(method.configure_optimizers()["optimizer"], torch.optim.Adam)


# -----------------------------------------------------------------------------
# Test suite for GenerativeFlow
# -----------------------------------------------------------------------------
class TestBaseGenerativeFlow:
    """Tests for GenerativeFlow matching and training logic."""

    def test_init_defaults(self, mock_dims_registry, mock_data_manager):
        """Default attributes should be None when not provided."""
        flow = DummyGenerativeFlow(mock_dims_registry, mock_data_manager)
        assert flow._probability_path is None
        assert flow._match_fn is None
        assert flow._noise_sampler is None
        assert flow._time_sampler is None
        assert flow.generate_from_noise is True

    def test_generate_from_noise_forced_when_unpaired(self, mock_dims_registry, mock_data_manager):
        """When the data manager is unpaired, generate_from_noise should be forced to True."""
        flow = DummyGenerativeFlow(
            mock_dims_registry,
            mock_data_manager,
            generate_from_noise=False,  # user tries to set False
        )
        assert flow.generate_from_noise is True

    def test_generate_from_noise_respected_when_paired(self, mock_dims_registry, mock_paired_data_manager):
        """When paired, generate_from_noise can be set to False."""
        flow = DummyGenerativeFlow(mock_dims_registry, mock_paired_data_manager, generate_from_noise=False)
        assert flow.generate_from_noise is False

    def test_properties_return_assigned_values(self, mock_dims_registry, mock_paired_data_manager):
        prob_path = Mock()
        match_fn = Mock()
        noise_sampler = Mock()
        time_sampler = Mock()
        flow = DummyGenerativeFlow(
            mock_dims_registry,
            mock_paired_data_manager,
            probability_path=prob_path,
            match_fn=match_fn,
            noise_sampler=noise_sampler,
            time_sampler=time_sampler,
            generate_from_noise=True,
        )
        assert flow.probability_path is prob_path
        assert flow.match_fn is match_fn
        assert flow.noise_sampler is noise_sampler
        assert flow.time_sampler is time_sampler
        assert flow.generate_from_noise is True

    def test_call_match_fn_safe_no_source(self, gen_flow):
        src_lin = src_quad = None
        tgt_lin = Mock()
        tgt_quad = Mock()
        src_idx, tgt_idx = gen_flow._call_match_fn_safe(src_lin, src_quad, tgt_lin, tgt_quad)
        assert src_idx is None
        assert tgt_idx is None
        gen_flow._match_fn.assert_not_called()

    def test_call_match_fn_safe_with_source(self, gen_flow):
        src_lin = torch.randn(4, 2)
        src_quad = None
        tgt_lin = torch.randn(4, 2)
        tgt_quad = None
        gen_flow._match_fn.return_value = (torch.tensor([0, 1]), torch.tensor([2, 3]))
        src_idx, tgt_idx = gen_flow._call_match_fn_safe(src_lin, src_quad, tgt_lin, tgt_quad)
        gen_flow._match_fn.assert_called_once_with(
            source_lin=src_lin, target_lin=tgt_lin, source_quad=src_quad, target_quad=tgt_quad
        )
        assert torch.equal(src_idx, torch.tensor([0, 1]))
        assert torch.equal(tgt_idx, torch.tensor([2, 3]))

    def test_match_observations_no_match(self, gen_flow):
        step_data = StepData(
            target_state=torch.randn(4, 2),
            target_coupling_lin=None,
            target_coupling_quad=None,
            target_condition_data={"a": torch.randn(4, 3)},
            target_group_data=None,
            source_state=torch.randn(4, 2),
            source_coupling_lin=None,
            source_coupling_quad=None,
            source_condition_data={"a": torch.randn(4, 3)},
            source_group_data=None,
        )
        gen_flow._match_fn = Mock(return_value=(None, None))
        new_step = gen_flow._match_observations(step_data)
        assert new_step.target_state is step_data.target_state
        assert new_step.source_state is step_data.source_state

    def test_match_observations_with_match(self, gen_flow):
        # Use tensors for condition_data (not dict) to allow indexing
        step_data = StepData(
            target_state=torch.randn(4, 2),
            target_coupling_lin=torch.randn(4, 2),
            target_coupling_quad=None,
            target_condition_data=torch.randn(4, 3),  # tensor
            target_group_data=None,
            source_state=torch.randn(4, 2),
            source_coupling_lin=torch.randn(4, 2),
            source_coupling_quad=None,
            source_condition_data=torch.randn(4, 3),  # tensor
            source_group_data=None,
        )
        src_idx = torch.tensor([0, 2])
        tgt_idx = torch.tensor([1, 3])
        gen_flow._match_fn = Mock(return_value=(src_idx, tgt_idx))

        new_step = gen_flow._match_observations(step_data)
        assert new_step.target_state.shape[0] == 2
        assert new_step.source_state.shape[0] == 2
        assert new_step.target_condition_data.shape[0] == 2
        assert new_step.source_condition_data.shape[0] == 2

    def test_train_step_forward(self, gen_flow):
        step_data = Mock()
        gen_flow.compute_loss = Mock(return_value=(torch.tensor(0.5), {"loss": 0.5}))
        loss, info = gen_flow._train_step_forward(step_data)
        assert loss == 0.5
        assert info["loss"] == 0.5

    def test_predict_with_no_grad(self, gen_flow):
        step_data = StepData(
            target_state=torch.randn(4, 2),
            target_coupling_lin=None,
            target_coupling_quad=None,
            target_condition_data=None,
            target_group_data=None,
            source_state=None,
            source_coupling_lin=None,
            source_coupling_quad=None,
            source_condition_data=None,
            source_group_data=None,
        )
        pred_data = PredictionData(X=torch.randn(4, 2), traj=None)
        gen_flow.infer = Mock(return_value=pred_data)
        result = gen_flow.predict(step_data, no_grad=True)
        assert result is pred_data
        gen_flow.infer.assert_called_once_with(step_data)

    def test_predict_rejects_wrong_type(self, gen_flow):
        with pytest.raises(ValueError, match="StepData"):
            gen_flow.predict(object())
