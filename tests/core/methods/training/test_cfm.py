from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import torch

from sckitflow._samplers import (
    IsotropicGaussianNoiseSamplerConfig,
    SamplerConfig,
    UniformTimeSamplerConfig,
)
from sckitflow.core._data_utils import prepare_latent_train
from sckitflow.core.methods.training._cfm import CFMTraining, CFMTrainingConfig
from sckitflow.core.probability_paths import (
    BasePath,
    LinearTwoSidedPath,
    PathConfig,
)
from sckitflow.data._dims import DataDimensions

MODULE = "sckitflow.core.methods.training._cfm"


# -------------------- Helpers and fixtures --------------------
class DummyModule(torch.nn.Module):
    """Velocity field stand-in that records its last call and returns zeros."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)
        self.last_call = None

    def forward(self, t, x, condition_dict=None, source=None):
        self.last_call = {"t": t, "x": x, "condition_dict": condition_dict, "source": source}
        return torch.zeros_like(x)


class LinearModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, t, x, condition_dict=None, source=None):
        return self.linear(x)


def dummy_probability_path() -> MagicMock:
    """A live ``BasePath``-shaped mock with fixed compute_xt/ut."""
    path = MagicMock(spec=BasePath)
    path.compute_xt.side_effect = lambda t, x0, x1, *, generator: torch.zeros_like(x0)
    path.compute_ut.side_effect = lambda t, xt, x0, x1: torch.ones_like(x0)
    return path


def _to_path_config(mock: MagicMock) -> PathConfig:
    """Wrap a live path mock in a config whose ``build()`` returns it.

    Subclasses a *concrete* path config (``LinearTwoSidedPath``)
    rather than the family base ``PathConfig``: the base runs a model
    validator that rejects non-member subclasses ("family base; give a spec with
    a 'type', or a member").  Subclassing the concrete member passes that check
    while still letting us override ``build()``.
    """

    class _MockPathConfig(LinearTwoSidedPath):
        def build(self):
            return mock

    return _MockPathConfig(sigma=1.0)


def _to_sampler_config(fn, *, concrete_cls):
    """Wrap a raw sampler callable in a config whose ``build()`` returns it.

    Like ``_to_path_config``, subclass the concrete sampler config so the
    ``SamplerConfig`` family validator accepts it.
    """

    class _CallableSamplerConfig(concrete_cls):
        def build(self):
            return fn

    return _CallableSamplerConfig()


def dummy_time_sampler(shape, *, generator, device=None, dtype=None):
    return torch.full(shape, 0.5, device=device, dtype=dtype)


def dummy_noise_sampler(shape, *, generator, device=None, dtype=None):
    return torch.rand(shape, generator=generator, device=device, dtype=dtype)


def gen(seed=0):
    return torch.Generator().manual_seed(seed)


@pytest.fixture
def dims() -> DataDimensions:
    return DataDimensions(
        state_dim=4,
        feature_names=pd.Index([f"f{i}" for i in range(4)]),
        condition_reps_dims={"c1": 3},
        condition_continuous_dims=None,
        response_reps_dims=None,
        response_continuous_dims=None,
        groups_reps_dims={"g1": 2},
        source_lin_dim=None,
        source_quad_dim=None,
        target_lin_dim=None,
        target_quad_dim=None,
    )


@pytest.fixture
def dummy_module():
    return DummyModule()


@pytest.fixture
def make_cfm(dummy_module, dims):
    """Build a ``CFMTraining`` whose path and samplers are always configs.

    Accepts live path mocks or raw sampler callables from the tests and wraps
    them into configs, so the config stays config-only at every construction
    site.
    """

    def _make(module=None, probability_path=None, time_sampler=None, noise_sampler=None, **overrides):
        # Path: live mock or config or missing -> config
        if probability_path is None:
            probability_path = LinearTwoSidedPath(sigma=1.0)
        elif not isinstance(probability_path, PathConfig):
            probability_path = _to_path_config(probability_path)

        # Time sampler: raw callable or config or default
        if time_sampler is None:
            time_sampler = UniformTimeSamplerConfig()
        elif not isinstance(time_sampler, SamplerConfig):
            time_sampler = _to_sampler_config(time_sampler, concrete_cls=UniformTimeSamplerConfig)

        # Noise sampler: same, but optional
        if noise_sampler is not None and not isinstance(noise_sampler, SamplerConfig):
            noise_sampler = _to_sampler_config(noise_sampler, concrete_cls=IsotropicGaussianNoiseSamplerConfig)

        kwargs = {
            "probability_path": probability_path,
            "time_sampler": time_sampler,
        }
        if noise_sampler is not None:
            kwargs["noise_sampler"] = noise_sampler
        kwargs.update(overrides)
        return CFMTraining(
            dummy_module if module is None else module,
            CFMTrainingConfig(**kwargs),
            dims=dims,
        )

    return _make


@pytest.fixture
def step_data():
    g = gen(1)
    return {
        "source_state": torch.randn(2, 4, generator=g),
        "target_state": torch.randn(2, 4, generator=g),
        "target_condition_data": {"c1": torch.randn(2, 3, generator=g)},
        "target_group_data": {"g1": torch.randn(2, 2, generator=g)},
    }


# -------------------- Construction --------------------
def test_default_config(dims):
    cfm = CFMTraining(DummyModule(), dims=dims)
    assert cfm.generate_from_noise is False
    assert cfm.probability_path is not None
    assert cfm.dims is dims


def test_build_from_config(dummy_module, dims):
    cfm = CFMTrainingConfig().build(dummy_module, dims=dims)
    assert isinstance(cfm, CFMTraining)
    assert cfm.module is dummy_module
    assert cfm.dims is dims


def test_placement_tracks_the_module(dims):
    module = DummyModule()
    cfm = CFMTraining(module, dims=dims)
    assert cfm.dtype is torch.float32
    module.double()
    assert cfm.dtype is torch.float64


# -------------------- Basic flow --------------------
def test_compute_loss_returns_scalar_and_metrics(make_cfm, step_data):
    loss, meta = make_cfm().compute_loss(step_data, generator=gen())
    assert loss.ndim == 0
    assert not meta["loss"].requires_grad and meta["loss"].item() == pytest.approx(
        loss.item()
    )  # a tensor: no sync per step


def test_compute_loss_calls_module_with_t_xt_cond_source(make_cfm, step_data, dummy_module):
    make_cfm().compute_loss(step_data, generator=gen())
    call = dummy_module.last_call
    assert call["t"].shape == (2,)
    assert call["x"].shape == (2, 4)
    assert set(call["condition_dict"]) == {"c1", "g1"}
    assert call["source"] is step_data["source_state"]


# -------------------- Source handling --------------------
def test_compute_loss_handles_none_source(make_cfm, dummy_module):
    step_data = {
        "source_state": None,
        "target_state": torch.randn(2, 4, generator=gen()),
        "target_condition_data": {"c1": torch.randn(2, 3, generator=gen())},
        "target_group_data": {},
    }
    cfm = make_cfm(generate_from_noise=True)
    loss, _ = cfm.compute_loss(step_data, generator=gen())
    assert torch.isfinite(loss)
    assert dummy_module.last_call["source"] is None


# -------------------- Conditioning --------------------
def test_compute_loss_merges_condition_and_group_data(make_cfm, step_data, dummy_module):
    make_cfm().compute_loss(step_data, generator=gen())
    cond = dummy_module.last_call["condition_dict"]
    torch.testing.assert_close(cond["c1"], step_data["target_condition_data"]["c1"])
    torch.testing.assert_close(cond["g1"], step_data["target_group_data"]["g1"])


# -------------------- Probability path --------------------
def test_probability_path_receives_latent_target_and_generator(make_cfm, step_data):
    path = dummy_probability_path()
    g = gen()
    make_cfm(probability_path=path).compute_loss(step_data, generator=g)

    xt_args = path.compute_xt.call_args.args
    assert [a.shape for a in xt_args] == [(2,), (2, 4), (2, 4)]
    assert path.compute_xt.call_args.kwargs["generator"] is g
    ut_args = path.compute_ut.call_args.args
    assert [a.shape for a in ut_args] == [(2,), (2, 4), (2, 4), (2, 4)]


# -------------------- Loss value --------------------
def test_loss_matches_mse_between_predicted_and_target_velocity(step_data, dims):
    class OnesModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 4)

        def forward(self, t, x, condition_dict=None, source=None):
            return torch.ones_like(x)

    path = dummy_probability_path()
    path.compute_ut.side_effect = lambda t, xt, x0, x1: 2 * torch.ones_like(x0)
    cfm = CFMTraining(
        OnesModule(),
        CFMTrainingConfig(
            probability_path=_to_path_config(path),
            time_sampler=UniformTimeSamplerConfig(),
        ),
        dims=dims,
    )
    loss, _ = cfm.compute_loss(step_data, generator=gen())
    assert loss.item() == pytest.approx(1.0)


# -------------------- prepare_latent_train wiring --------------------
def test_prepare_latent_train_called_with_dims_placement_and_flags(make_cfm, step_data, dims, dummy_module):
    cfm = make_cfm(generate_from_noise=True)
    g = gen()
    with patch(f"{MODULE}.prepare_latent_train") as mock_prep:
        mock_prep.return_value = torch.zeros(2, dims.state_dim)
        cfm.compute_loss(step_data, generator=g)

    args, kwargs = mock_prep.call_args
    # Positional: source, noise_sampler, dims; the batch size is the target's.
    assert args[0] is step_data["source_state"]
    assert args[1] is cfm.noise_sampler
    assert args[2] is dims
    assert kwargs["batch_size"] == step_data["target_state"].shape[0]
    # Keyword: placement, flags, generator.
    first_param = next(dummy_module.parameters())
    assert kwargs["device"] == first_param.device
    assert kwargs["dtype"] == first_param.dtype
    assert kwargs["generate_from_noise"] is True
    assert kwargs["generator"] is g


# -------------------- dtype --------------------
def test_loss_dtype_follows_batch(make_cfm, step_data):
    step_data = {
        k: {kk: vv.double() for kk, vv in v.items()} if isinstance(v, dict) else v.double()
        for k, v in step_data.items()
    }
    loss, _ = make_cfm().compute_loss(step_data, generator=gen())
    assert loss.dtype == torch.float64


# -------------------- Randomness --------------------
@pytest.mark.parametrize("generate_from_noise", [False, True])
def test_compute_loss_depends_only_on_generator(step_data, dims, generate_from_noise):
    cfm = CFMTraining(
        LinearModule(),
        CFMTrainingConfig(
            probability_path=LinearTwoSidedPath(sigma=0.1),
            generate_from_noise=generate_from_noise,
        ),
        dims=dims,
    )

    def loss(seed, global_seed):
        torch.manual_seed(global_seed)
        return cfm.compute_loss(step_data, generator=gen(seed))[0]

    torch.testing.assert_close(loss(0, 1), loss(0, 2))
    assert not torch.allclose(loss(0, 1), loss(1, 1))


# -------------------- dims is what sizes the training latent --------------------
def test_train_latent_uses_dims_state_dim():
    """A wrong-width source must not change the generated latent's width -- `dims.state_dim` is the width.

    Uses a wide `dims` and a deliberately wrong-width source, and asserts the
    latent comes out at `dims.state_dim` in the generating branch. If the
    refactor is ever reverted to reading width from a batch tensor, this fails on
    shape.
    """
    wide = DataDimensions(
        state_dim=11,
        feature_names=pd.Index([f"f{i}" for i in range(11)]),
        condition_reps_dims=None,
        condition_continuous_dims=None,
        response_reps_dims=None,
        response_continuous_dims=None,
        groups_reps_dims=None,
        source_lin_dim=None,
        source_quad_dim=None,
        target_lin_dim=None,
        target_quad_dim=None,
    )
    source = torch.randn(3, 999)  # wrong on purpose
    latent = prepare_latent_train(
        source,
        IsotropicGaussianNoiseSamplerConfig().build(),
        wide,
        batch_size=3,
        device=torch.device("cpu"),
        dtype=torch.float32,
        generate_from_noise=True,
        generator=gen(),
    )
    assert latent.shape == (3, wide.state_dim)
