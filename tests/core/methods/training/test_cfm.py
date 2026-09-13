from unittest.mock import MagicMock, patch

import pytest
import torch

from sckitflow.core.methods.training._cfm import CFMTrainingProtocol

# Adjust the module path above to wherever CFMTrainingProtocol actually lives.
MODULE = "sckitflow.core.methods.training._cfm"


# -------------------- Helpers and fixtures --------------------
class DummyModule(torch.nn.Module):
    """Velocity field stand-in; returns a tensor of the same shape as x."""

    def __init__(self, shape=(2, 4)):
        super().__init__()
        self._shape = shape
        self.last_call = None

    def forward(self, t, x, condition_dict=None, source=None):
        self.last_call = {
            "t": t,
            "x": x,
            "condition_dict": condition_dict,
            "source": source,
        }
        return torch.zeros_like(x)


def dummy_probability_path():
    """Probability path stand-in returning zero interpolant/velocity."""
    path = MagicMock(name="probability_path")

    def compute_xt(t, x0, x1):
        return torch.zeros_like(x0)

    def compute_ut(t, xt, x0, x1):
        return torch.ones_like(x0)

    path.compute_xt.side_effect = compute_xt
    path.compute_ut.side_effect = compute_ut
    return path


def dummy_time_sampler(shape, device=None, dtype=None):
    return torch.full(shape, 0.5, device=device, dtype=dtype)


def dummy_noise_sampler(shape, device=None, dtype=None):
    return torch.randn(shape, device=device, dtype=dtype)


@pytest.fixture
def dummy_module():
    return DummyModule(shape=(2, 4))


@pytest.fixture
def make_cfm(dummy_module):
    """Build a CFMTrainingProtocol; override any kwarg via ``_make``."""

    def _make(**overrides):
        kwargs = {
            "module": dummy_module,
            "probability_path": dummy_probability_path(),
            "time_sampler": dummy_time_sampler,
            "device_id": "cpu",
        }
        kwargs.update(overrides)
        return CFMTrainingProtocol(**kwargs)

    return _make


@pytest.fixture
def step_data():
    return {
        "source_state": torch.randn(2, 4),
        "target_state": torch.randn(2, 4),
        "target_condition_data": {"c1": torch.randn(2, 3)},
        "target_group_data": {"g1": torch.randn(2, 2)},
    }


# -------------------- Basic flow --------------------
def test_compute_loss_returns_scalar_and_metrics(make_cfm, step_data):
    cfm = make_cfm()
    loss, meta = cfm.compute_loss(step_data)

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert isinstance(meta, dict)
    assert "loss" in meta
    assert meta["loss"] == pytest.approx(loss.item())


def test_compute_loss_calls_module_with_t_xt_cond_source(make_cfm, step_data, dummy_module):
    cfm = make_cfm()
    cfm.compute_loss(step_data)

    call = dummy_module.last_call
    # Time samples shaped (batch_size,)
    assert call["t"].shape == (2,)
    # Interpolated state has the same shape as the latent (and target)
    assert call["x"].shape == (2, 4)
    # Conditioning dict carries both condition and group entries
    assert set(call["condition_dict"].keys()) == {"c1", "g1"}
    # Source forwarded as passed in
    assert call["source"] is not None
    assert call["source"].shape == (2, 4)


# -------------------- Source handling --------------------
def test_compute_loss_handles_none_source(make_cfm, dummy_module):
    step_data = {
        "source_state": None,
        "target_state": torch.randn(2, 4),
        "target_condition_data": {},
        "target_group_data": {},
    }
    cfm = make_cfm(
        noise_sampler=dummy_noise_sampler,
        generate_from_noise=True,
    )

    loss, _ = cfm.compute_loss(step_data)
    assert torch.isfinite(loss)
    # The module receives ``source=None`` unchanged.
    assert dummy_module.last_call["source"] is None


def test_compute_loss_coerces_source_device_dtype(make_cfm, step_data, dummy_module):
    cfm = make_cfm(dtype=torch.float64)
    # Replace source with an int64 tensor to prove coercion.
    step_data["source_state"] = torch.ones(2, 4, dtype=torch.int64)

    cfm.compute_loss(step_data)

    assert dummy_module.last_call["source"].dtype == torch.float64


# -------------------- Conditioning --------------------
def test_compute_loss_merges_condition_and_group_data(make_cfm, step_data, dummy_module):
    cfm = make_cfm()
    cfm.compute_loss(step_data)

    cond = dummy_module.last_call["condition_dict"]
    torch.testing.assert_close(cond["c1"], step_data["target_condition_data"]["c1"])
    torch.testing.assert_close(cond["g1"], step_data["target_group_data"]["g1"])


def test_compute_loss_coerces_conditioning_device_dtype(make_cfm, step_data, dummy_module):
    cfm = make_cfm(dtype=torch.float64)
    step_data["target_condition_data"] = {"c1": torch.ones(2, 3, dtype=torch.int64)}
    step_data["target_group_data"] = {"g1": torch.ones(2, 2, dtype=torch.int64)}

    cfm.compute_loss(step_data)

    cond = dummy_module.last_call["condition_dict"]
    assert cond["c1"].dtype == torch.float64
    assert cond["g1"].dtype == torch.float64


# -------------------- Time sampling --------------------
def test_time_sampler_called_with_batch_shape_and_latent_device_dtype(make_cfm, step_data):
    cfm = make_cfm(dtype=torch.float64)
    # Instrument the sampler directly; the property returns the stored callable.
    cfm._time_sampler = MagicMock(wraps=dummy_time_sampler)
    cfm.compute_loss(step_data)

    args, kwargs = cfm._time_sampler.call_args
    shape = args[0]
    assert tuple(shape) == (2,)
    assert kwargs["device"] == torch.device("cpu")
    assert kwargs["dtype"] == torch.float64


# -------------------- Probability path --------------------
def test_probability_path_receives_latent_and_target(make_cfm, step_data):
    path = dummy_probability_path()
    cfm = make_cfm(probability_path=path)

    cfm.compute_loss(step_data)

    # compute_xt(t, x0, x1)
    xt_args = path.compute_xt.call_args.args
    assert xt_args[0].shape == (2,)  # t
    assert xt_args[1].shape == (2, 4)  # latent
    assert xt_args[2].shape == (2, 4)  # target

    # compute_ut(t, xt, x0, x1)
    ut_args = path.compute_ut.call_args.args
    assert ut_args[0].shape == (2,)
    assert ut_args[1].shape == (2, 4)  # xt
    assert ut_args[2].shape == (2, 4)  # latent
    assert ut_args[3].shape == (2, 4)  # target


# -------------------- Loss value --------------------
def test_loss_matches_mse_between_predicted_and_target_velocity(dummy_module, step_data):
    # Arrange a module that returns a known constant; a path that returns
    # a known constant target velocity -> loss has a predictable value.
    class KnownModule(torch.nn.Module):
        def forward(self, t, x, condition_dict=None, source=None):
            return torch.ones_like(x)

    path = MagicMock()
    path.compute_xt.side_effect = lambda t, x0, x1: torch.zeros_like(x0)
    # Target velocity is 2x ones -> (vt - ut)^2 == 1 everywhere.
    path.compute_ut.side_effect = lambda t, xt, x0, x1: 2 * torch.ones_like(x0)

    cfm = CFMTrainingProtocol(
        module=KnownModule(),
        probability_path=path,
        time_sampler=dummy_time_sampler,
        device_id="cpu",
    )
    loss, _ = cfm.compute_loss(step_data)

    assert loss.item() == pytest.approx(1.0)


# -------------------- Integration with prepare_latent_train --------------------
def test_latent_shape_matches_target_shape(make_cfm, step_data, dummy_module):
    cfm = make_cfm()
    cfm.compute_loss(step_data)

    # t is sampled with the latent's batch size; the module's x has the
    # latent shape; both should be (2, 4) for this fixture.
    assert dummy_module.last_call["x"].shape == (2, 4)


def test_prepare_latent_train_called_with_flow_flags(make_cfm, step_data):
    cfm = make_cfm(
        noise_sampler=dummy_noise_sampler,
        generate_from_noise=True,
    )

    with patch(f"{MODULE}.prepare_latent_train", wraps=None) as mock_prep:
        mock_prep.return_value = torch.randn(2, 4)
        cfm.compute_loss(step_data)

    kwargs = mock_prep.call_args.kwargs
    assert kwargs["generate_from_noise"] is True
    # source, target, noise_sampler are positional
    args = mock_prep.call_args.args
    assert args[0] is step_data["source_state"]
    assert args[1] is step_data["target_state"]
    assert args[2] is cfm.noise_sampler


# -------------------- dtype propagation --------------------
def test_loss_dtype_matches_protocol_dtype(make_cfm, step_data):
    cfm = make_cfm(dtype=torch.float64)
    loss, _ = cfm.compute_loss(step_data)
    assert loss.dtype == torch.float64
