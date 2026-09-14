from unittest.mock import MagicMock, patch

import pytest
import torch

from sckitflow.core._types import PredictionData
from sckitflow.core.methods._base import FlowSpecs
from sckitflow.core.methods.inference._ode import ODEInference

# Adjust the module path above to wherever ODEInference lives.
# The patch targets below assume the imports are resolved inside that module.


# -------------------- Fixtures and helpers --------------------
MODULE = "sckitflow.core.methods.inference._ode"


class DummyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, t, x, **kwargs):
        return self.linear(x)

    def get_vf_fn(self, *args, **kwargs):
        return self.forward


def dummy_probability_path():
    return MagicMock(name="probability_path")


def dummy_time_sampler(shape, device=None, dtype=None):
    return torch.rand(shape, device=device, dtype=dtype)


def dummy_noise_sampler(shape, device=None, dtype=None):
    return torch.randn(shape, device=device, dtype=dtype)


@pytest.fixture
def dummy_module():
    return DummyModule()


@pytest.fixture
def step_data():
    return {
        "source_state": torch.randn(2, 4),
        "target_state": torch.randn(2, 4),
        "target_condition_data": {"cond": torch.randn(2, 3)},
        "target_group_data": {"group": torch.randn(2, 2)},
    }


@pytest.fixture
def make_inference(dummy_module):
    """Build an ODEInference with sensible defaults; override any kwarg."""

    def _make(ode_kwargs: dict | None = None, **overrides):
        kwargs = {
            "module": dummy_module,
            "probability_path": dummy_probability_path(),
            "time_sampler": dummy_time_sampler,
            "device_id": "cpu",
        }
        kwargs.update(overrides)
        specs = FlowSpecs(**kwargs)
        ode_kwargs = {} if ode_kwargs is None else ode_kwargs
        return ODEInference(specs, **ode_kwargs)

    return _make


# -------------------- Construction and properties --------------------
def test_init_stores_flow_specs_and_extras(make_inference, dummy_module):
    inference = make_inference(
        ode_kwargs={
            "solver_kwargs": {"rtol": 1e-5},
            "return_trajectory": True,
            "n_steps": 42,
            "n_samples": 7,
        }
    )

    # FlowSpecs inherited
    assert inference.module is dummy_module
    assert inference.device_id == "cpu"
    assert inference.dtype == torch.float32
    assert inference.generate_from_noise is False
    assert inference.noise_sampler is torch.randn

    # Extras
    assert inference.solver_kwargs == {"rtol": 1e-5}
    assert inference.return_trajectory is True
    assert inference.n_steps == 42
    assert inference.n_samples == 7
    assert inference.latent is None


def test_init_forwards_flow_specs_to_parent(make_inference):
    ns = dummy_noise_sampler
    inference = make_inference(
        noise_sampler=ns,
        generate_from_noise=True,
        dtype=torch.float64,
    )
    assert inference.noise_sampler is ns
    assert inference.generate_from_noise is True
    assert inference.dtype == torch.float64


# -------------------- Guards --------------------
def test_predict_raises_when_generating_from_noise_without_n_samples(make_inference, step_data):
    inference = make_inference(
        ode_kwargs={
            "n_samples": None,
        },
        noise_sampler=dummy_noise_sampler,
        generate_from_noise=True,
    )
    with pytest.raises(ValueError, match="number of samples"):
        inference.predict(step_data)


# -------------------- Latent preparation --------------------
def test_predict_uses_user_supplied_latent(make_inference, step_data):
    supplied = torch.randn(5, 4)
    inference = make_inference(ode_kwargs={"latent": supplied, "n_samples": 3})

    with (
        patch(f"{MODULE}.prepare_latent_inference") as mock_prep,
        patch(f"{MODULE}.ODESolver") as mock_solver_cls,
        patch(f"{MODULE}.aggregate_predictions") as mock_agg,
        patch(f"{MODULE}.expand_conditioning") as mock_expand,
    ):
        mock_expand.return_value = ({}, torch.randn(5, 4))
        mock_solver = mock_solver_cls.return_value
        mock_solver.solve.return_value = torch.randn(5, 4)
        mock_agg.return_value = (
            torch.randn(5, 4),
            None,
            torch.randn(5, 4),
        )

        inference.predict(step_data)

    # The user-supplied latent bypasses prepare_latent_inference entirely.
    mock_prep.assert_not_called()
    # And lands in the solver.
    args, _ = mock_solver.solve.call_args
    assert args[0].shape == (5, 4)


def test_predict_prepares_latent_from_step_data(make_inference, step_data):
    inference = make_inference(
        ode_kwargs={
            "n_samples": 4,
        },
        generate_from_noise=True,
        noise_sampler=dummy_noise_sampler,
    )
    fake_latent = torch.randn(4, 2, 4)

    with (
        patch(f"{MODULE}.prepare_latent_inference") as mock_prep,
        patch(f"{MODULE}.ODESolver") as mock_solver_cls,
        patch(f"{MODULE}.aggregate_predictions") as mock_agg,
        patch(f"{MODULE}.expand_conditioning") as mock_expand,
    ):
        mock_prep.return_value = fake_latent
        mock_expand.return_value = ({}, fake_latent)
        mock_solver = mock_solver_cls.return_value
        mock_solver.solve.return_value = fake_latent
        mock_agg.return_value = (fake_latent, None, None)

        inference.predict(step_data)

    mock_prep.assert_called_once()
    kwargs = mock_prep.call_args.kwargs
    assert kwargs["n_samples"] == 4
    assert kwargs["generate_from_noise"] is True


# -------------------- Solver configuration --------------------
def test_solver_uses_default_method_when_not_specified(make_inference, step_data):
    inference = make_inference(ode_kwargs={"latent": torch.randn(2, 4)})

    with (
        patch(f"{MODULE}.prepare_latent_inference"),
        patch(f"{MODULE}.ODESolver") as mock_solver_cls,
        patch(f"{MODULE}.aggregate_predictions") as mock_agg,
        patch(f"{MODULE}.expand_conditioning") as mock_expand,
    ):
        mock_expand.return_value = ({}, torch.randn(2, 4))
        mock_solver_cls.return_value.solve.return_value = torch.randn(2, 4)
        mock_agg.return_value = (torch.randn(2, 4), None, None)

        inference.predict(step_data)

    init_kwargs = mock_solver_cls.call_args.kwargs
    assert init_kwargs["method"] == "euler"


def test_solver_uses_supplied_method(make_inference, step_data):
    inference = make_inference(
        ode_kwargs={"latent": torch.randn(2, 4), "solver_kwargs": {"method": "rk4", "rtol": 1e-6}}
    )

    with (
        patch(f"{MODULE}.prepare_latent_inference"),
        patch(f"{MODULE}.ODESolver") as mock_solver_cls,
        patch(f"{MODULE}.aggregate_predictions") as mock_agg,
        patch(f"{MODULE}.expand_conditioning") as mock_expand,
    ):
        mock_expand.return_value = ({}, torch.randn(2, 4))
        mock_solver_cls.return_value.solve.return_value = torch.randn(2, 4)
        mock_agg.return_value = (torch.randn(2, 4), None, None)

        inference.predict(step_data)

    init_kwargs = mock_solver_cls.call_args.kwargs
    solve_kwargs = mock_solver_cls.return_value.solve.call_args.kwargs

    assert init_kwargs["method"] == "rk4"
    # "method" was popped; the rest is forwarded to solve.
    assert "method" not in solve_kwargs["solver_kwargs"]
    assert solve_kwargs["solver_kwargs"]["rtol"] == 1e-6


def test_solver_does_not_mutate_user_solver_kwargs(make_inference, step_data):
    user_kwargs = {"method": "rk4", "rtol": 1e-6}
    inference = make_inference(ode_kwargs={"latent": torch.randn(2, 4), "solver_kwargs": user_kwargs})

    with (
        patch(f"{MODULE}.prepare_latent_inference"),
        patch(f"{MODULE}.ODESolver") as mock_solver_cls,
        patch(f"{MODULE}.aggregate_predictions") as mock_agg,
        patch(f"{MODULE}.expand_conditioning") as mock_expand,
    ):
        mock_expand.return_value = ({}, torch.randn(2, 4))
        mock_solver_cls.return_value.solve.return_value = torch.randn(2, 4)
        mock_agg.return_value = (torch.randn(2, 4), None, None)

        inference.predict(step_data)

    # The user's dict is untouched: no "method" popped.
    assert user_kwargs == {"method": "rk4", "rtol": 1e-6}


def test_time_grid_uses_n_steps_and_latent_properties(make_inference, step_data):
    latent = torch.randn(2, 4)
    inference = make_inference(ode_kwargs={"latent": latent, "n_steps": 7})

    with (
        patch(f"{MODULE}.prepare_latent_inference"),
        patch(f"{MODULE}.ODESolver") as mock_solver_cls,
        patch(f"{MODULE}.aggregate_predictions") as mock_agg,
        patch(f"{MODULE}.expand_conditioning") as mock_expand,
        patch(f"{MODULE}.torch.linspace", wraps=torch.linspace) as mock_linspace,
    ):
        mock_expand.return_value = ({}, latent)
        mock_solver_cls.return_value.solve.return_value = latent
        mock_agg.return_value = (latent, None, None)

        inference.predict(step_data)

    kwargs = mock_linspace.call_args.kwargs
    assert kwargs["steps"] == 7
    assert kwargs["device"] == latent.device
    assert kwargs["dtype"] == latent.dtype


# -------------------- Trajectory forwarding --------------------
def test_return_trajectory_is_forwarded_to_solve_and_aggregate(make_inference, step_data):
    inference = make_inference(ode_kwargs={"latent": torch.randn(2, 4), "return_trajectory": True})

    with (
        patch(f"{MODULE}.prepare_latent_inference"),
        patch(f"{MODULE}.ODESolver") as mock_solver_cls,
        patch(f"{MODULE}.aggregate_predictions") as mock_agg,
        patch(f"{MODULE}.expand_conditioning") as mock_expand,
    ):
        mock_expand.return_value = ({}, torch.randn(2, 4))
        mock_solver_cls.return_value.solve.return_value = torch.randn(5, 2, 4)
        mock_agg.return_value = (torch.randn(2, 4), torch.randn(5, 2, 4), None)

        inference.predict(step_data)

    assert mock_solver_cls.return_value.solve.call_args.kwargs["return_trajectory"] is True
    assert mock_agg.call_args.kwargs["return_trajectory"] is True


# -------------------- Integration (no external mocks beyond the solver) --------------------
def test_predict_returns_prediction_data_with_expected_fields(make_inference, step_data):
    inference = make_inference(ode_kwargs={"latent": torch.randn(2, 4)})

    with (
        patch(f"{MODULE}.prepare_latent_inference"),
        patch(f"{MODULE}.ODESolver") as mock_solver_cls,
        patch(f"{MODULE}.aggregate_predictions") as mock_agg,
        patch(f"{MODULE}.expand_conditioning") as mock_expand,
    ):
        X = torch.randn(2, 4)
        traj = torch.randn(10, 2, 4)
        raw = torch.randn(2, 4)

        mock_expand.return_value = ({}, X)
        mock_solver_cls.return_value.solve.return_value = traj
        mock_agg.return_value = (X, traj, raw)

        result = inference.predict(step_data)

    assert isinstance(result, PredictionData)
    torch.testing.assert_close(result.X, X)
    torch.testing.assert_close(result.traj, traj)
    torch.testing.assert_close(result.raw_samples, raw)


def test_predict_passes_condition_dict_and_source_to_solver(make_inference, step_data):
    inference = make_inference(ode_kwargs={"latent": torch.randn(2, 4)})
    cond = {"cond": torch.randn(2, 3)}
    source = torch.randn(2, 4)

    with (
        patch(f"{MODULE}.prepare_latent_inference"),
        patch(f"{MODULE}.ODESolver") as mock_solver_cls,
        patch(f"{MODULE}.aggregate_predictions") as mock_agg,
        patch(f"{MODULE}.expand_conditioning") as mock_expand,
        patch(f"{MODULE}.get_tensor_dict_from_data") as mock_get,
    ):
        mock_get.side_effect = [
            {"cond": torch.randn(2, 3)},
            {"group": torch.randn(2, 2)},
        ]
        mock_expand.return_value = (cond, source)
        mock_solver_cls.return_value.solve.return_value = torch.randn(2, 4)
        mock_agg.return_value = (torch.randn(2, 4), None, None)

        inference.predict(step_data)

    init_kwargs = mock_solver_cls.call_args.kwargs
    assert init_kwargs["vf_kwargs"]["condition_dict"] is cond
    assert init_kwargs["vf_kwargs"]["source"] is source
