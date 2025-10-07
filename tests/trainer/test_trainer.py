import pytest
import numpy as np
import matplotlib.pyplot as plt
from sc_flow.trainer._trainer import FlowTrainer
from .utils import get_dummy_network
import torch

input_dim = 10
output_dim = 10
hidden_dims = (20, 20)
batch_size = 32


@pytest.fixture
def dummy_method(input_di=input_dim, output_dim=output_dim, hidden_dims=hidden_dims):
    return get_dummy_network(input_dim, output_dim, hidden_dims)


@pytest.fixture
def dummy_callbacks():
    class DummyCallbacks:
        def __init__(self):
            self.called_with = []

        def run_on_valid_step(self, validation_dict, condition):
            self.called_with.append((validation_dict, condition))
            return {"val_loss": 0.456}

    return DummyCallbacks()


@pytest.fixture
def trainer(dummy_method, dummy_callbacks):
    return FlowTrainer(method=dummy_method, require_prng=False, callbacks=dummy_callbacks)


# ----------------------------------------------------------------------
# __validation_step and _validation_step
# ----------------------------------------------------------------------
"""
def test___validation_step_returns_expected_dict(trainer):
    batch = {"target": 1, "input": 2}
    result = trainer._FlowTrainer__validation_step(batch)
    assert result == {"target": 1, "prediction": {"pred": 2}}
    assert trainer._method.eval_called == 1
"""

def test__validation_step_runs_callbacks(trainer):cd 
    val_data = {"condA": {"source": torch.rand((batch_size, output_dim)), 
                            "target": torch.rand((batch_size, output_dim))}}
    metrics = trainer._validation_step(val_data)
    assert metrics == {"val_loss": 0.456}
    #assert trainer._method.eval_called == 1
    assert trainer._callbacks.called_with[0][1] == "condA"


# ----------------------------------------------------------------------
# _train_step
# ----------------------------------------------------------------------

def test__train_step_calls_train_and_returns_loss(trainer):
    batch = {"source": torch.rand((batch_size, output_dim)), 
                    "target": torch.rand((batch_size, output_dim))}
    loss = trainer._train_step(batch)
    assert trainer._method.train_called == 1
    assert isinstance(loss, torch.Tensor)


# ----------------------------------------------------------------------
# __update_logs
# ----------------------------------------------------------------------

def test___update_logs_appends_metrics(trainer):
    trainer._FlowTrainer__update_logs({"acc": 0.9, "loss": 0.5})
    assert trainer._training_logs["acc"] == [0.9]
    assert trainer._training_logs["loss"][-1] == 0.5


# ----------------------------------------------------------------------
# fit
# ----------------------------------------------------------------------

@pytest.fixture
def dummy_trainloader():
    class DummyTrainLoader:
        def __init__(self):
            self.sample_calls = 0

        def sample(self, _):
            self.sample_calls += 1
            return {"source": torch.rand((batch_size, output_dim)), 
                    "target": torch.rand((batch_size, output_dim))}

    return DummyTrainLoader()

@pytest.fixture
def dummy_valloader():
    class DummyValLoader:
        def __init__(self):
            self.sample_calls = 0

        def sample(self, _):
            self.sample_calls += 1
            return {"condA": {"source": torch.rand((batch_size, output_dim)), 
                    "target": torch.rand((batch_size, output_dim))}}

    return DummyValLoader()


def test_fit_runs_basic_training_loop(monkeypatch, trainer, dummy_trainloader, dummy_valloader):

    monkeypatch.setattr("sc_flow._runtime.BACKEND", "torch")

    monkeypatch.setitem(trainer.__dict__, "_require_prng", False)
    trainer.fit(train_dataloader=dummy_trainloader, num_iterations=3, valid_freq=1, validation_dataloader=dummy_valloader)
    # loss logged each iteration
    print(trainer._training_logs["loss"])
    assert len(trainer._training_logs["loss"]) == 3
    # validation step performed
    assert dummy_trainloader.sample_calls >= 1


def test_fit_warns_if_prng_provided_but_not_required(monkeypatch, trainer, dummy_trainloader, dummy_valloader, caplog):
    monkeypatch.setattr("sc_flow._runtime.BACKEND", "torch")
    trainer.fit(train_dataloader=dummy_trainloader, num_iterations=1, valid_freq=1, validation_dataloader=dummy_valloader, prng="FAKE")
    assert any("PRNG provided" in m for m in caplog.text.splitlines())


# ----------------------------------------------------------------------
# plot_training_logs
# ----------------------------------------------------------------------

def test_plot_training_logs_single_key(trainer):
    fig, ax = trainer.plot_training_logs(keys_to_plot="loss")
    assert isinstance(fig, plt.Figure)
    # axes may be one Axes object
    assert hasattr(ax, "plot")


def test_plot_training_logs_multiple_keys(trainer):
    trainer._training_logs["val_loss"] = [0.1, 0.2]
    fig, axes = trainer.plot_training_logs(keys_to_plot=["loss", "val_loss"])
    assert len(axes) == 2


def test_plot_training_logs_missing_key_raises(trainer):
    with pytest.raises(AssertionError):
        trainer.plot_training_logs(keys_to_plot="nonexistent")