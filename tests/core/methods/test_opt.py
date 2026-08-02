import pytest
import torch

from sckitflow.core.methods._opt import OptimConfig
from sckitflow.core.nn._modules import BaseModule


# -----------------------------------------------------------------------------
# Dummy module for testing (simple linear layer)
# -----------------------------------------------------------------------------
class DummyModule(BaseModule):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)

    def _make_modules(self):
        pass  # required by BaseModule but not needed

    def forward(self, x):
        return self.linear(x)


# -----------------------------------------------------------------------------
# Test suite for OptimConfig
# -----------------------------------------------------------------------------
class TestOptimConfigOptimizer:
    """Tests for how `OptimConfig` resolves the optimizer."""

    @pytest.mark.slow
    def test_pre_created_optimizer_is_used_as_is(self):
        module = DummyModule()
        optimizer = torch.optim.SGD(module.parameters(), lr=0.01)
        config = OptimConfig(optimizer=optimizer)
        assert config.build_optimizer(module.parameters()) is optimizer

    def test_string_optimizer(self):
        module = DummyModule()
        config = OptimConfig(optimizer_cls="Adam", lr=1e-3, optimizer_kwargs={"weight_decay": 0.01})
        optimizer = config.build_optimizer(module.parameters())
        assert isinstance(optimizer, torch.optim.Adam)
        assert optimizer.defaults["lr"] == 1e-3
        assert optimizer.defaults["weight_decay"] == 0.01

    def test_callable_optimizer(self):
        module = DummyModule()
        config = OptimConfig(optimizer_cls=torch.optim.AdamW, lr=2e-4)
        optimizer = config.build_optimizer(module.parameters())
        assert isinstance(optimizer, torch.optim.AdamW)
        assert optimizer.defaults["lr"] == 2e-4

    def test_defaults_to_adam(self):
        module = DummyModule()
        assert isinstance(OptimConfig().build_optimizer(module.parameters()), torch.optim.Adam)

    def test_unknown_optimizer_name_raises(self):
        module = DummyModule()
        config = OptimConfig(optimizer_cls="NotAnOptimizer")
        with pytest.raises(ValueError, match="not found in torch.optim"):
            config.build_optimizer(module.parameters())

    def test_non_callable_optimizer_cls_raises(self):
        module = DummyModule()
        config = OptimConfig(optimizer_cls=42)
        with pytest.raises(TypeError, match="must be a string or callable"):
            config.build_optimizer(module.parameters())


class TestOptimConfigScheduler:
    """Tests for how `OptimConfig` resolves the learning rate scheduler."""

    def test_no_scheduler_by_default(self):
        module = DummyModule()
        config = OptimConfig()
        assert config.build_lr_scheduler(config.build_optimizer(module.parameters())) is None

    def test_pre_created_scheduler_is_used_as_is(self):
        module = DummyModule()
        optimizer = torch.optim.Adam(module.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
        config = OptimConfig(optimizer=optimizer, lr_scheduler=scheduler)
        assert config.build_lr_scheduler(optimizer) is scheduler

    def test_string_scheduler_is_attached_to_the_optimizer(self):
        module = DummyModule()
        config = OptimConfig(
            optimizer_cls="Adam", lr=0.001, lr_scheduler_cls="StepLR", lr_scheduler_kwargs={"step_size": 10}
        )
        optimizer = config.build_optimizer(module.parameters())
        scheduler = config.build_lr_scheduler(optimizer)
        assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)
        assert scheduler.optimizer is optimizer

    def test_unknown_scheduler_name_raises(self):
        module = DummyModule()
        config = OptimConfig(lr_scheduler_cls="NotAScheduler")
        with pytest.raises(ValueError, match="not found in torch.optim.lr_scheduler"):
            config.build_lr_scheduler(config.build_optimizer(module.parameters()))


class TestOptimConfigResolve:
    """Tests for the `configure_optimizers` mapping that Lightning consumes."""

    def test_resolve_without_scheduler(self):
        module = DummyModule()
        resolved = OptimConfig(optimizer_cls="SGD", lr=0.1).resolve(module.parameters())
        assert set(resolved) == {"optimizer"}
        assert isinstance(resolved["optimizer"], torch.optim.SGD)

    def test_resolve_with_scheduler_sets_step_interval(self):
        module = DummyModule()
        config = OptimConfig(lr_scheduler_cls="StepLR", lr_scheduler_kwargs={"step_size": 1})
        resolved = config.resolve(module.parameters())
        assert isinstance(resolved["lr_scheduler"]["scheduler"], torch.optim.lr_scheduler.StepLR)
        # Node-steps are the unit of progress, so the scheduler ticks per step.
        assert resolved["lr_scheduler"]["interval"] == "step"

    @pytest.mark.parametrize(
        ("configured", "expected"),
        [("train_step", "step"), ("step", "step"), ("epoch", "epoch")],
    )
    def test_scheduler_interval_aliases(self, configured, expected):
        """`"train_step"` is the historical spelling of Lightning's `"step"`."""
        assert OptimConfig(lr_scheduler_step=configured).lr_scheduler_interval == expected

    def test_invalid_scheduler_interval_raises(self):
        with pytest.raises(ValueError, match="lr_scheduler_step must be one of"):
            OptimConfig(lr_scheduler_step="every_other_tuesday")

    def test_monitor_is_omitted_unless_requested(self):
        """Lightning rejects a `None` monitor, so the key must be absent by default."""
        module = DummyModule()
        config = OptimConfig(lr_scheduler_cls="StepLR", lr_scheduler_kwargs={"step_size": 1})
        assert "monitor" not in config.resolve(module.parameters())["lr_scheduler"]

    def test_monitor_is_forwarded_when_set(self):
        module = DummyModule()
        config = OptimConfig(
            lr_scheduler_cls="ReduceLROnPlateau",
            lr_scheduler_monitor="valA_mmd",
        )
        assert config.resolve(module.parameters())["lr_scheduler"]["monitor"] == "valA_mmd"
