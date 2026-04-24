import torch

from sc_flow.backends.torch.methods._opt import TorchOptimizationManager
from sc_flow.backends.torch.nn._modules import BaseModule
from sc_flow.methods._opt import OptimConfig


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
# Test suite for TorchOptimizationManager
# -----------------------------------------------------------------------------
class TestTorchOptimizationManager:
    """Tests for the TorchOptimizationManager class."""

    def test_from_config_with_pre_created_optimizer(self):
        module = DummyModule()
        optimizer = torch.optim.SGD(module.parameters(), lr=0.01)
        config = OptimConfig(optimizer=optimizer)
        manager = TorchOptimizationManager.from_config(module, config)
        assert manager._optimizer is optimizer

    def test_from_config_with_string_optimizer(self):
        module = DummyModule()
        config = OptimConfig(optimizer_cls="Adam", lr=1e-3, optimizer_kwargs={"weight_decay": 0.01})
        manager = TorchOptimizationManager.from_config(module, config)
        assert isinstance(manager._optimizer, torch.optim.Adam)
        assert manager._optimizer.defaults["lr"] == 1e-3
        assert manager._optimizer.defaults["weight_decay"] == 0.01

    def test_from_config_with_callable_optimizer(self):
        module = DummyModule()
        config = OptimConfig(optimizer_cls=torch.optim.AdamW, lr=2e-4)
        manager = TorchOptimizationManager.from_config(module, config)
        assert isinstance(manager._optimizer, torch.optim.AdamW)
        assert manager._optimizer.defaults["lr"] == 2e-4

    def test_from_config_with_pre_created_scheduler(self):
        module = DummyModule()
        optimizer = torch.optim.Adam(module.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
        config = OptimConfig(optimizer=optimizer, lr_scheduler=scheduler)
        manager = TorchOptimizationManager.from_config(module, config)
        assert manager._lr_scheduler is scheduler

    def test_from_config_with_string_scheduler(self):
        module = DummyModule()
        config = OptimConfig(
            optimizer_cls="Adam", lr=0.001, lr_scheduler_cls="StepLR", lr_scheduler_kwargs={"step_size": 10}
        )
        manager = TorchOptimizationManager.from_config(module, config)
        assert isinstance(manager._lr_scheduler, torch.optim.lr_scheduler.StepLR)
        # Check that scheduler is attached to the optimizer
        assert manager._lr_scheduler.optimizer is manager._optimizer

    def test_step(self):
        module = DummyModule()
        optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
        manager = TorchOptimizationManager(optimizer)
        x = torch.randn(2, 2)
        loss = module(x).sum()
        # Before step, gradients are None
        for param in module.parameters():
            assert param.grad is None
        manager.step(loss)
        # After backward, gradients exist
        for param in module.parameters():
            assert param.grad is not None

    def test_step_with_scheduler(self):
        module = DummyModule()
        optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        manager = TorchOptimizationManager(optimizer, lr_scheduler=scheduler, lr_scheduler_step="train_step")
        x = torch.randn(2, 2)
        loss = module(x).sum()
        # Store initial learning rate
        initial_lr = optimizer.param_groups[0]["lr"]
        manager.step(loss)
        # After one step, scheduler should have updated lr (since step_size=1)
        assert optimizer.param_groups[0]["lr"] != initial_lr
