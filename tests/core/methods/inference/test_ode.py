from unittest.mock import patch

import pytest
import torch
from pydantic import ValidationError

from sckitflow.core._types import PredictionData
from sckitflow.core.methods.inference._ode import ODEInference, ODEInferenceConfig

MODULE = "sckitflow.core.methods.inference._ode"


# -------------------- Fixtures and helpers --------------------
class DummyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, t, x, **kwargs):
        return self.linear(x)

    def get_vf_fn(self, *args, **kwargs):
        return self.forward


def dummy_time_sampler(shape, *, generator, device=None, dtype=None):
    return torch.rand(shape, generator=generator, device=device, dtype=dtype)


def dummy_noise_sampler(shape, *, generator, device=None, dtype=None):
    return torch.randn(shape, generator=generator, device=device, dtype=dtype)


def gen(seed=0):
    return torch.Generator().manual_seed(seed)


@pytest.fixture
def dummy_module():
    return DummyModule()


@pytest.fixture
def step_data():
    g = gen(1)
    return {
        "source_state": torch.randn(2, 4, generator=g),
        "target_state": torch.randn(2, 4, generator=g),
        "target_condition_data": {"cond": torch.randn(2, 3, generator=g)},
        "target_group_data": {"group": torch.randn(2, 2, generator=g)},
    }


@pytest.fixture
def make_inference(dummy_module):
    def _make(latent=None, **config):
        config.setdefault("time_sampler", dummy_time_sampler)
        return ODEInference(dummy_module, ODEInferenceConfig(**config), latent=latent)

    return _make


@pytest.fixture
def mocks():
    """Patches latent preparation, conditioning expansion, the solver and aggregation."""
    with (
        patch(f"{MODULE}.prepare_latent_inference") as prep,
        patch(f"{MODULE}.ODESolver") as solver_cls,
        patch(f"{MODULE}.aggregate_predictions") as agg,
        patch(f"{MODULE}.expand_conditioning") as expand,
    ):
        expand.return_value = ({}, torch.zeros(2, 4))
        solver_cls.return_value.solve.return_value = torch.zeros(2, 4)
        agg.return_value = (torch.zeros(2, 4), None, None)
        yield {"prep": prep, "solver_cls": solver_cls, "agg": agg, "expand": expand}


# -------------------- Construction and config --------------------
def test_init_stores_module_and_config(make_inference, dummy_module):
    inference = make_inference(solver_kwargs={"rtol": 1e-5}, return_trajectory=True, n_steps=42, n_samples=7)

    assert inference.module is dummy_module
    assert inference.generate_from_noise is False
    assert inference.latent is None
    assert inference.config.solver_kwargs == {"rtol": 1e-5}
    assert inference.config.return_trajectory is True
    assert inference.config.n_steps == 42
    assert inference.config.n_samples == 7


def test_init_forwards_flow_config_to_parent(make_inference):
    inference = make_inference(noise_sampler=dummy_noise_sampler, generate_from_noise=True, n_samples=2)
    assert inference.noise_sampler is dummy_noise_sampler
    assert inference.time_sampler is dummy_time_sampler
    assert inference.generate_from_noise is True


def test_default_config(dummy_module):
    inference = ODEInference(dummy_module)
    assert inference.config == ODEInferenceConfig()
    assert ODEInferenceConfig().build(dummy_module).module is dummy_module


# -------------------- Validation --------------------
def test_generate_from_noise_requires_n_samples():
    with pytest.raises(ValidationError, match="n_samples"):
        ODEInferenceConfig(generate_from_noise=True)


def test_n_steps_must_be_at_least_two():
    with pytest.raises(ValidationError):
        ODEInferenceConfig(n_steps=1)


# -------------------- Latent preparation --------------------
def test_predict_uses_user_supplied_latent(make_inference, step_data, mocks):
    make_inference(latent=torch.randn(5, 4, generator=gen()), n_samples=3).predict(step_data, generator=gen())

    mocks["prep"].assert_not_called()
    assert mocks["solver_cls"].return_value.solve.call_args.args[0].shape == (5, 4)


def test_predict_prepares_latent_from_step_data(make_inference, step_data, mocks):
    inference = make_inference(n_samples=4, generate_from_noise=True, noise_sampler=dummy_noise_sampler)
    g = gen()
    mocks["prep"].return_value = torch.zeros(4, 2, 4)
    inference.predict(step_data, generator=g)

    mocks["prep"].assert_called_once()
    kwargs = mocks["prep"].call_args.kwargs
    assert kwargs["n_samples"] == 4
    assert kwargs["generate_from_noise"] is True
    assert kwargs["generator"] is g


# -------------------- Solver configuration --------------------
def test_solver_uses_default_method_when_not_specified(make_inference, step_data, mocks):
    make_inference(latent=torch.zeros(2, 4)).predict(step_data, generator=gen())
    assert mocks["solver_cls"].call_args.kwargs["method"] == "euler"


def test_solver_uses_supplied_method(make_inference, step_data, mocks):
    make_inference(latent=torch.zeros(2, 4), solver_kwargs={"method": "rk4", "rtol": 1e-6}).predict(
        step_data, generator=gen()
    )
    solve_kwargs = mocks["solver_cls"].return_value.solve.call_args.kwargs["solver_kwargs"]
    assert mocks["solver_cls"].call_args.kwargs["method"] == "rk4"
    assert solve_kwargs == {"rtol": 1e-6}


def test_predict_does_not_mutate_config_solver_kwargs(make_inference, step_data, mocks):
    inference = make_inference(latent=torch.zeros(2, 4), solver_kwargs={"method": "rk4", "rtol": 1e-6})
    inference.predict(step_data, generator=gen())
    inference.predict(step_data, generator=gen())
    assert inference.config.solver_kwargs == {"method": "rk4", "rtol": 1e-6}
    assert mocks["solver_cls"].call_args.kwargs["method"] == "rk4"


def test_time_grid_uses_n_steps_and_latent_properties(make_inference, step_data, mocks):
    latent = torch.zeros(2, 4, dtype=torch.float64)
    with patch(f"{MODULE}.torch.linspace", wraps=torch.linspace) as mock_linspace:
        make_inference(latent=latent, n_steps=7).predict(step_data, generator=gen())

    kwargs = mock_linspace.call_args.kwargs
    assert kwargs["steps"] == 7
    assert kwargs["device"] == latent.device
    # the latent is cast to the batch dtype before the grid is built
    assert kwargs["dtype"] == step_data["target_state"].dtype


# -------------------- Trajectory forwarding --------------------
def test_return_trajectory_is_forwarded_to_solve_and_aggregate(make_inference, step_data, mocks):
    make_inference(latent=torch.zeros(2, 4), return_trajectory=True).predict(step_data, generator=gen())
    assert mocks["solver_cls"].return_value.solve.call_args.kwargs["return_trajectory"] is True
    assert mocks["agg"].call_args.kwargs["return_trajectory"] is True


# -------------------- Outputs and wiring --------------------
def test_predict_returns_prediction_data_with_expected_fields(make_inference, step_data, mocks):
    X, traj, raw = torch.zeros(2, 4), torch.ones(10, 2, 4), torch.full((2, 4), 2.0)
    mocks["agg"].return_value = (X, traj, raw)
    result = make_inference(latent=torch.zeros(2, 4)).predict(step_data, generator=gen())

    assert isinstance(result, PredictionData)
    assert result.X is X
    assert result.traj is traj
    assert result.raw_samples is raw


def test_predict_passes_condition_dict_and_source_to_solver(make_inference, step_data, mocks):
    cond, source = {"cond": torch.zeros(2, 3)}, torch.zeros(2, 4)
    mocks["expand"].return_value = (cond, source)
    make_inference(latent=torch.zeros(2, 4)).predict(step_data, generator=gen())

    expand_cond = mocks["expand"].call_args.args[1]
    assert set(expand_cond) == {"cond", "group"}
    vf_kwargs = mocks["solver_cls"].call_args.kwargs["vf_kwargs"]
    assert vf_kwargs["condition_dict"] is cond
    assert vf_kwargs["source"] is source


# -------------------- Randomness --------------------
def test_predict_depends_only_on_generator(make_inference, step_data):
    inference = make_inference(generate_from_noise=True, n_samples=3, n_steps=5)

    def predict(seed, global_seed):
        torch.manual_seed(global_seed)
        return inference.predict(step_data, generator=gen(seed)).X

    torch.testing.assert_close(predict(0, 1), predict(0, 2))
    assert not torch.allclose(predict(0, 1), predict(1, 1))


def test_solver_kwargs_may_set_the_tolerances(make_inference, step_data):
    inference = make_inference(n_steps=5, solver_kwargs={"method": "dopri5", "rtol": 1e-4, "atol": 1e-5})
    assert inference.predict(step_data, generator=gen(0)).X.shape == step_data["target_state"].shape
