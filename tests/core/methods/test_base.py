import pandas as pd
import pydantic
import pytest
import torch

from sckitflow._samplers import (
    IsotropicGaussianNoiseSamplerConfig,
    TimeSamplerConfig,
    UniformTimeSamplerConfig,
    _standard_normal,
    _uniform,
)
from sckitflow.core._types import new_step_data
from sckitflow.core.methods._base import (
    AbstractFlowMethod,
    AbstractMethod,
    BaseMatcher,
    FlowMethodConfig,
    MatchedTrainingMethod,
    Matcher,
    SupportsInference,
    SupportsTraining,
    _placement_of,
)
from sckitflow.core.methods.training._cfm import CFMTrainingConfig
from sckitflow.core.probability_paths._config import LinearTwoSidedPathConfig
from sckitflow.core.probability_paths._probability_paths import LinearTwoSidedPath
from sckitflow.data._dims import DataDimensions


@pytest.fixture
def dims() -> DataDimensions:
    return DataDimensions(
        state_dim=4,
        feature_names=pd.Index([f"f{i}" for i in range(4)]),
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


@pytest.fixture
def module() -> torch.nn.Module:
    return torch.nn.Linear(4, 4)


def my_sampler(shape, *, generator, device=None, dtype=None):
    return torch.zeros(shape, device=device, dtype=dtype)


class _MySamplerConfig(TimeSamplerConfig):
    """Tiny config that binds the module-level ``my_sampler`` for the tests."""

    fn = my_sampler


class Trainer:
    """Structural `SupportsTraining` -- carries `dims`, `device`, `dtype`."""

    def __init__(self, module, dims):
        self.module = module
        self.dims = dims
        self.device = next(iter(module.parameters())).device
        self.dtype = next(iter(module.parameters())).dtype
        self.calls = []

    def compute_loss(self, step_data, *, generator):
        self.calls.append((step_data, generator))
        return torch.tensor(1.0), {}


class Predictor:
    """Structural `SupportsInference` -- carries `dims`, `device`, `dtype`."""

    def __init__(self, module, dims):
        self.module = module
        self.dims = dims
        self.device = next(iter(module.parameters())).device
        self.dtype = next(iter(module.parameters())).dtype

    def predict(self, step_data, *, generator):
        return None


@pytest.fixture
def coupled():
    g = torch.Generator().manual_seed(0)
    return new_step_data(
        source_state=torch.arange(3.0)[:, None],
        target_state=torch.arange(3.0, 6.0)[:, None],
        source_coupling_lin=torch.randn(3, 2, generator=g),
        target_coupling_lin=torch.randn(3, 2, generator=g),
    )


# -------------------- _placement_of --------------------
def test_placement_of_reads_the_first_parameter():
    module = torch.nn.Linear(2, 2).double()
    device, dtype = _placement_of(module)
    assert device.type == "cpu"
    assert dtype is torch.float64


def test_placement_of_defaults_on_a_parameterless_module():
    """A module with nothing to move cannot be on a GPU; CPU/float32 is the truth, not a fallback."""
    device, dtype = _placement_of(torch.nn.Identity())
    assert device.type == "cpu"
    assert dtype is torch.float32


# -------------------- AbstractMethod --------------------
def test_abstract_method_keeps_module_as_given(dims):
    module = torch.nn.Linear(4, 4).double()
    method = AbstractMethod(module, dims=dims)
    assert method.module is module
    assert next(method.module.parameters()).dtype == torch.float64
    assert next(method.module.parameters()).device.type == "cpu"


def test_abstract_method_keeps_dims_by_identity(module, dims):
    method = AbstractMethod(module, dims=dims)
    assert method.dims is dims


def test_abstract_method_device_and_dtype_track_the_module(module, dims):
    """Placement is resolved per call, so a moved module is picked up on the next read."""
    method = AbstractMethod(module, dims=dims)
    assert method.device.type == "cpu"
    assert method.dtype is torch.float32

    module.double()  # simulates a precision move
    assert method.dtype is torch.float64


# -------------------- AbstractFlowMethod --------------------
def test_flow_method_defaults(module, dims):
    method = AbstractFlowMethod(module, CFMTrainingConfig(), dims=dims)
    assert isinstance(method.probability_path, LinearTwoSidedPath)
    assert method.time_sampler is _uniform
    assert method.noise_sampler is _standard_normal
    assert method.generate_from_noise is False


def test_flow_method_builds_path_config(module, dims):
    config = CFMTrainingConfig(probability_path=LinearTwoSidedPathConfig(sigma=0.5))
    method = AbstractFlowMethod(module, config, dims=dims)
    assert isinstance(method.probability_path, LinearTwoSidedPath)


def test_flow_method_custom_values(module, dims):
    """Non-default configs are built; the method stores their live objects."""
    path_cfg = LinearTwoSidedPathConfig(sigma=0.5)
    config = CFMTrainingConfig(
        probability_path=path_cfg,
        time_sampler=UniformTimeSamplerConfig(),
        noise_sampler=IsotropicGaussianNoiseSamplerConfig(),
        generate_from_noise=True,
    )
    method = AbstractFlowMethod(module, config, dims=dims)
    assert isinstance(method.probability_path, LinearTwoSidedPath)
    assert method.generate_from_noise is True


def test_flow_method_accepts_sampler_config(module, dims):
    """A ``SamplerConfig`` is resolved to its underlying callable on the method."""
    config = CFMTrainingConfig(
        time_sampler=UniformTimeSamplerConfig(),
        noise_sampler=IsotropicGaussianNoiseSamplerConfig(),
    )
    method = AbstractFlowMethod(module, config, dims=dims)
    assert method.time_sampler is _uniform
    assert method.noise_sampler is _standard_normal


def test_flow_method_keeps_dims_by_identity(module, dims):
    method = AbstractFlowMethod(module, CFMTrainingConfig(), dims=dims)
    assert method.dims is dims


@pytest.mark.parametrize("sampler", [_uniform, _standard_normal])
def test_default_samplers_draw_from_generator(sampler):
    a = sampler((5,), generator=torch.Generator().manual_seed(0))
    torch.manual_seed(123)  # the global RNG must not matter
    b = sampler((5,), generator=torch.Generator().manual_seed(0))
    torch.testing.assert_close(a, b)
    assert not torch.equal(a, sampler((5,), generator=torch.Generator().manual_seed(1)))


# -------------------- FlowMethodConfig --------------------
def test_family_base_config_is_not_buildable():
    with pytest.raises(pydantic.ValidationError, match="family base"):
        FlowMethodConfig()


def test_config_rejects_non_sampler():
    """A raw int is neither a ``SamplerConfig`` nor ``None`` -> validation fails."""
    with pytest.raises(pydantic.ValidationError):
        CFMTrainingConfig(time_sampler=3)


def test_config_rejects_raw_callable_sampler():
    """A raw Python function is not a ``SamplerConfig`` -> validation fails.

    The contract is now configs-only: ``Model.save_pretrained`` must not receive a spec
    that can't be JSON-serialized, so the rejection happens at construction
    time instead of at ``to_spec`` time.
    """
    with pytest.raises(pydantic.ValidationError):
        CFMTrainingConfig(time_sampler=my_sampler)


def test_config_rejects_live_probability_path():
    """Live ``BaseProbabilityPath`` instances are rejected; only configs are accepted."""
    with pytest.raises(pydantic.ValidationError):
        CFMTrainingConfig(probability_path=LinearTwoSidedPath(sigma=0.5))


def test_config_rejects_swapped_samplers():
    """A noise sampler cannot fill the time slot, nor the other way round."""
    with pytest.raises(pydantic.ValidationError):
        CFMTrainingConfig(time_sampler=IsotropicGaussianNoiseSamplerConfig())
    with pytest.raises(pydantic.ValidationError):
        CFMTrainingConfig(noise_sampler=UniformTimeSamplerConfig())


def test_config_with_sampler_config_is_portable():
    """A ``SamplerConfig`` serializes cleanly and round-trips."""
    config = CFMTrainingConfig(
        time_sampler=UniformTimeSamplerConfig(),
        noise_sampler=IsotropicGaussianNoiseSamplerConfig(),
    )
    spec = config.to_spec()
    assert "time_sampler" in spec
    assert "noise_sampler" in spec
    assert CFMTrainingConfig.from_spec(spec) == config


def test_cfm_config_spec_round_trips():
    config = CFMTrainingConfig()
    assert CFMTrainingConfig.from_spec(config.to_spec()) == config


# -------------------- Protocols --------------------
def test_structural_protocols(module, dims):
    assert isinstance(Trainer(module, dims), SupportsTraining)
    assert not isinstance(Trainer(module, dims), SupportsInference)
    assert isinstance(Predictor(module, dims), SupportsInference)
    assert not isinstance(Predictor(module, dims), SupportsTraining)


def test_protocol_requires_dims(module, dims):
    """A method without `dims` is not `SupportsTraining` / `SupportsInference`."""

    class NoDims:
        @property
        def module(self):
            return module

        def compute_loss(self, step_data, *, generator):
            return torch.tensor(0.0), {}

    assert not isinstance(NoDims(), SupportsTraining)
    assert not isinstance(NoDims(), SupportsInference)


# -------------------- Matcher --------------------
def test_base_matcher_is_abstract():
    with pytest.raises(TypeError):
        BaseMatcher(match_fn=lambda **_: (None, None))


def test_match_unchanged_without_source_coupling():
    def match_fn(**_):
        raise AssertionError("match_fn must not be called")

    data = new_step_data(target_coupling_lin=torch.zeros(3, 2))
    assert Matcher(match_fn).match(data, generator=torch.Generator()) == (data, {})


def test_match_unchanged_when_indices_none(coupled):
    assert Matcher(lambda **_: (None, None)).match(coupled, generator=torch.Generator()) == (coupled, {})


def test_match_subscripts_and_passes_generator(coupled):
    generator = torch.Generator()
    seen = {}

    def match_fn(source_lin, target_lin, source_quad, target_quad, generator):
        seen.update(source_lin=source_lin, target_lin=target_lin, generator=generator)
        return torch.tensor([2, 0]), torch.tensor([1, 1])

    out, metrics = Matcher(match_fn).match(coupled, generator=generator)
    assert metrics == {}  # this match_fn returns no plan
    assert seen["generator"] is generator
    assert seen["source_lin"] is coupled["source_coupling_lin"]
    assert seen["target_lin"] is coupled["target_coupling_lin"]
    torch.testing.assert_close(out["source_state"], torch.tensor([[2.0], [0.0]]))
    torch.testing.assert_close(out["target_state"], torch.tensor([[4.0], [4.0]]))


# -------------------- MatchedTrainingMethod --------------------
def test_matched_training_method(module, dims, coupled):
    inner = Trainer(module, dims)
    matcher = Matcher(lambda **_: (torch.tensor([1]), torch.tensor([2])))
    matched = MatchedTrainingMethod(inner, matcher)
    generator = torch.Generator().manual_seed(0)

    loss, _ = matched.compute_loss(coupled, generator=generator)

    assert loss.item() == 1.0
    ((data, seen_generator),) = inner.calls
    assert seen_generator is generator
    torch.testing.assert_close(data["source_state"], torch.tensor([[1.0]]))
    torch.testing.assert_close(data["target_state"], torch.tensor([[5.0]]))
    assert matched.method is inner
    assert matched.matcher is matcher
    assert matched.module is module
    assert isinstance(matched, SupportsTraining)


def test_matched_training_method_forwards_dims(module, dims, coupled):
    inner = Trainer(module, dims)
    matched = MatchedTrainingMethod(inner, Matcher(lambda **_: (None, None)))
    assert matched.dims is dims
    assert isinstance(matched, SupportsTraining)


def test_matched_training_method_forwards_placement(module, dims, coupled):
    """`device` / `dtype` are what a downstream consumer of the wrapper would reach for."""
    module.double()
    inner = Trainer(module, dims)
    matched = MatchedTrainingMethod(inner, Matcher(lambda **_: (None, None)))
    assert matched.dtype is torch.float64
    assert matched.device.type == "cpu"


@pytest.mark.parametrize(("plan", "fallback"), [(torch.full((3, 3), 1 / 9), 0.0), (torch.zeros(3, 3), 1.0)])
def test_match_logs_when_the_plan_falls_back(coupled, plan, fallback):
    def match_fn(*, source_lin, target_lin, source_quad, target_quad, generator, return_matrix=False):
        idxs = torch.tensor([0, 1]), torch.tensor([1, 0])
        return (*idxs, plan) if return_matrix else idxs

    _, metrics = Matcher(match_fn).match(coupled, generator=torch.Generator())
    assert isinstance(metrics["coupling_fallback"], torch.Tensor) and metrics["coupling_fallback"].item() == fallback
