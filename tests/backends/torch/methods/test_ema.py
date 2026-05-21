import pytest
import torch
import torch.nn as nn

from sc_flow.backends.torch.methods._ema import ExponentialMovingAverage


# -----------------------------------------------------------------------------
# Dummy model for testing
# -----------------------------------------------------------------------------
class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(5, 2)
        self.bias = nn.Parameter(torch.zeros(2))

    def forward(self, x):
        return self.linear(x) + self.bias


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture
def model():
    return DummyModel()


@pytest.fixture
def ema(model):
    return ExponentialMovingAverage(model, decay=0.9)


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
class TestExponentialMovingAverage:
    def test_init_registers_shadow_params(self, model, ema):
        """Shadow parameters should be created for trainable parameters."""
        trainable_params = [name for name, p in model.named_parameters() if p.requires_grad]
        assert set(ema._shadow_params.keys()) == set(trainable_params)
        for name in trainable_params:
            assert torch.equal(ema._shadow_params[name], model.get_parameter(name).data)

    def test_update_formula(self, model, ema):
        """EMA update: shadow = decay * shadow + (1-decay) * param."""
        # Store initial shadow values
        initial_shadows = {name: ema._shadow_params[name].clone() for name in ema._shadow_params}
        # Modify model parameters (simulate training step)
        with torch.no_grad():
            for param in model.parameters():
                if param.requires_grad:
                    param.add_(torch.randn_like(param))
        ema.update()
        # Verify update
        for name, param in model.named_parameters():
            if param.requires_grad:
                expected = 0.9 * initial_shadows[name] + 0.1 * param.data
                assert torch.allclose(ema._shadow_params[name], expected)

    def test_copy_to(self, model, ema):
        """Copy EMA parameters to a model."""
        # Modify model parameters to some values
        with torch.no_grad():
            for param in model.parameters():
                if param.requires_grad:
                    param.fill_(1.0)
        # Update EMA (should still be zeros initially)
        ema.update()  # after one update, EMA shadow becomes 0.9*0 + 0.1*1 = 0.1
        # Create a copy model and copy EMA weights
        copy_model = DummyModel()
        ema.copy_to(copy_model)
        # Check that copy_model's parameters equal EMA shadows
        for name, param in copy_model.named_parameters():
            if param.requires_grad:
                assert torch.allclose(param, ema._shadow_params[name])

    def test_store_and_restore(self, model, ema):
        """Store current model params, modify, then restore."""
        # Store initial state
        stored = ema.store()
        # Modify model
        with torch.no_grad():
            for param in model.parameters():
                if param.requires_grad:
                    param.fill_(42.0)
        # Restore
        ema.restore(stored)
        # Verify restoration
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert torch.allclose(param, stored[name])

    def test_context_manager(self, model, ema):
        """Context manager should temporarily use EMA weights for forward."""
        # Zero out shadow parameters to start from zero
        for name in ema._shadow_params:
            ema._shadow_params[name].zero_()
        # Set model weights to 1.0
        with torch.no_grad():
            for param in model.parameters():
                if param.requires_grad:
                    param.fill_(1.0)
        # Update EMA once so shadows become 0.1 (since shadows were zero)
        ema.update()
        # Store original model weights for later check
        original_weights = ema.store()
        # Inside context, model should have EMA weights
        with ema as ema_instance:
            assert ema_instance is ema
            # Model parameters should now be EMA shadows (0.1)
            for param in model.parameters():
                if param.requires_grad:
                    assert torch.allclose(param, torch.full_like(param, 0.1))
        # After context, model should be restored to original weights (1.0)
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert torch.allclose(param, original_weights[name])

    def test_multiple_updates(self, model, ema):
        """Check cumulative effect of multiple updates."""
        # Zero out shadow parameters to start from zero
        for name in ema._shadow_params:
            ema._shadow_params[name].zero_()
        # Set model to constant 1.0
        with torch.no_grad():
            for param in model.parameters():
                if param.requires_grad:
                    param.fill_(1.0)
        # After 10 updates, shadow should approach 1.0
        for _ in range(10):
            ema.update()
        # Expected value after 10 updates with decay 0.9: 1 - 0.9^10 = 0.6513...
        expected = 1.0 - 0.9**10
        for name in ema._shadow_params:
            assert torch.allclose(
                ema._shadow_params[name], torch.full_like(ema._shadow_params[name], expected), atol=1e-6
            )

    def test_state_dict_save_load(self, model, ema):
        """EMA state dict should be saveable and loadable."""
        # Perform some updates
        for _ in range(5):
            with torch.no_grad():
                for param in model.parameters():
                    if param.requires_grad:
                        param.add_(torch.randn_like(param))
            ema.update()
        # Save state dict
        state = ema.state_dict()
        # Create new EMA instance and load
        new_ema = ExponentialMovingAverage(model, decay=0.9)
        new_ema.load_state_dict(state)
        # Compare shadow params
        assert torch.allclose(ema._shadow_params["linear.weight"], new_ema._shadow_params["linear.weight"])
        assert ema._decay == new_ema._decay

    def test_to_device(self, model):
        """EMA should move shadow parameters to the specified device."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        ema = ExponentialMovingAverage(model, decay=0.9)
        device = torch.device("cuda:0")
        ema.to(device)
        for shadow in ema._shadow_params.values():
            assert shadow.device == device

    def test_no_trainable_params(self):
        """Model with no trainable parameters should still work (empty EMA)."""

        class NoTrainableModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("buf", torch.ones(1))

        model = NoTrainableModel()
        ema = ExponentialMovingAverage(model, decay=0.9)
        assert len(ema._shadow_params) == 0
        ema.update()  # should not error
        ema.copy_to(model)  # should not error
        state = ema.state_dict()
        assert state["shadow_params"] == {}

    def test_copy_to_partial_model(self, model, ema):
        """If target model has extra parameters, copy_to should ignore them."""

        # Create a model with an extra parameter
        class ExtendedModel(DummyModel):
            def __init__(self):
                super().__init__()
                self.extra = nn.Parameter(torch.zeros(1))

        extended = ExtendedModel()
        # Copy EMA from original model to extended model
        ema.copy_to(extended)
        # Check that common parameters were copied
        for name in ema._shadow_params:
            assert torch.allclose(extended.get_parameter(name).data, ema._shadow_params[name])
        # Extra parameter should remain unchanged
        assert torch.allclose(extended.extra, torch.zeros(1))
