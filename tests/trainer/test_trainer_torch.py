import matplotlib.pyplot as plt
import pytest
import torch

from sc_flow.trainer._trainer import FlowTrainer

input_dim = 10
output_dim = 10
hidden_dims = (20, 20)
batch_size = 32


@pytest.fixture
def trainer(dummy_method, dummy_callbacks):
    return FlowTrainer(method=dummy_method, require_prng=False, callbacks=dummy_callbacks)


# ----------------------------------------------------------------------
# __validation_step and _validation_step
# ----------------------------------------------------------------------


def test__validation_step_on_single_conditionreturns_expected_dict(trainer):
    batch = {"target": 1, "input": 2}
    result = trainer._validation_step_on_single_condition(batch)
    assert result == {"target": 1, "prediction": None}
    assert trainer._method.eval_called == 1


def test__validation_step_runs_callbacks(trainer):
    val_data = {
        "condA": {"source": torch.rand((batch_size, output_dim)), "target": torch.rand((batch_size, output_dim))}
    }
    metrics = trainer._validation_step(val_data)
    assert isinstance(metrics, dict)
    assert trainer._callbacks.called_with[0][1] == "condA"


# ----------------------------------------------------------------------
# _train_step
# ----------------------------------------------------------------------


def test__train_step_calls_train_and_returns_loss(trainer):
    batch = {"source": torch.rand((batch_size, output_dim)), "target": torch.rand((batch_size, output_dim))}
    loss = trainer._train_step(batch)
    assert trainer._method.train_called == 1
    assert isinstance(loss, torch.Tensor)


# ----------------------------------------------------------------------
# _update_logs
# ----------------------------------------------------------------------


def test__update_logs_appends_metrics(trainer):
    trainer._update_logs({"acc": 0.9, "loss": 0.5})
    assert trainer._training_logs["acc"] == [0.9]
    assert trainer._training_logs["loss"][-1] == 0.5


# ----------------------------------------------------------------------
# fit
# ----------------------------------------------------------------------


def test_fit_runs_basic_training_loop(monkeypatch, trainer, dummy_trainloader_torch, dummy_valloader_torch):
    monkeypatch.setattr("sc_flow._runtime.BACKEND", "torch")
    monkeypatch.setitem(trainer.__dict__, "_require_prng", False)
    trainer.fit(
        train_dataloader=dummy_trainloader_torch,
        num_iterations=3,
        valid_freq=1,
        validation_dataloader=dummy_valloader_torch,
    )
    # loss logged each iteration
    print(trainer._training_logs["loss"])
    assert len(trainer._training_logs["loss"]) == 3
    # validation step performed
    assert dummy_trainloader_torch.sample_calls >= 1


def test_fit_warns_if_prng_provided_but_not_required(
    monkeypatch, trainer, dummy_trainloader_torch, dummy_valloader_torch, caplog
):
    monkeypatch.setattr("sc_flow._runtime.BACKEND", "torch")
    trainer.fit(
        train_dataloader=dummy_trainloader_torch,
        num_iterations=1,
        valid_freq=1,
        validation_dataloader=dummy_valloader_torch,
        prng="FAKE",
    )
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
