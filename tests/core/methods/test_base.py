import numpy as np
import pydantic
import pytest
import torch
from scfit.registry import PortabilityError

from sckitflow.core._types import new_step_data
from sckitflow.core.methods._base import (
    AbstractFlowMethod,
    AbstractFlowMethodConfig,
    AbstractMethod,
    BaseMatcher,
    MatchedTrainingMethod,
    Matcher,
    SupportsInference,
    SupportsTraining,
    _standard_normal,
    _uniform,
)
from sckitflow.core.methods.training._cfm import CFMTrainingConfig
from sckitflow.core.probability_paths._config import LinearGaussianProbabilityPathConfig
from sckitflow.core.probability_paths._probability_paths import (
    LinearDiracProbabilityPath,
    LinearGaussianProbabilityPath,
)


def my_sampler(shape, *, generator, device=None, dtype=None):
    return torch.zeros(shape, device=device, dtype=dtype)


class Trainer:
    def __init__(self, module):
        self.module = module
        self.calls = []

    def compute_loss(self, step_data, *, generator, rng):
        self.calls.append((step_data, generator, rng))
        return torch.tensor(1.0), {}


class Predictor:
    def __init__(self, module):
        self.module = module

    def predict(self, step_data, *, generator):
        return None


@pytest.fixture
def module():
    return torch.nn.Linear(4, 4)


@pytest.fixture
def coupled():
    g = torch.Generator().manual_seed(0)
    return new_step_data(
        source_state=torch.arange(3.0)[:, None],
        target_state=torch.arange(3.0, 6.0)[:, None],
        source_coupling_lin=torch.randn(3, 2, generator=g),
        target_coupling_lin=torch.randn(3, 2, generator=g),
    )


# -------------------- AbstractMethod --------------------
def test_abstract_method_keeps_module_as_given():
    module = torch.nn.Linear(4, 4).double()
    method = AbstractMethod(module)
    assert method.module is module
    assert next(method.module.parameters()).dtype == torch.float64
    assert next(method.module.parameters()).device.type == "cpu"


# -------------------- AbstractFlowMethod --------------------
def test_flow_method_defaults(module):
    method = AbstractFlowMethod(module, CFMTrainingConfig())
    assert isinstance(method.probability_path, LinearDiracProbabilityPath)
    assert method.time_sampler is _uniform
    assert method.noise_sampler is _standard_normal
    assert method.generate_from_noise is False


def test_flow_method_builds_path_config(module):
    config = CFMTrainingConfig(probability_path=LinearGaussianProbabilityPathConfig(sigma=0.5))
    assert isinstance(AbstractFlowMethod(module, config).probability_path, LinearGaussianProbabilityPath)


def test_flow_method_custom_values(module):
    path = LinearGaussianProbabilityPath(sigma=0.5)
    config = CFMTrainingConfig(
        probability_path=path, time_sampler=my_sampler, noise_sampler=my_sampler, generate_from_noise=True
    )
    method = AbstractFlowMethod(module, config)
    assert method.probability_path is path
    assert method.time_sampler is my_sampler
    assert method.noise_sampler is my_sampler
    assert method.generate_from_noise is True


@pytest.mark.parametrize("sampler", [_uniform, _standard_normal])
def test_default_samplers_draw_from_generator(sampler):
    a = sampler((5,), generator=torch.Generator().manual_seed(0))
    torch.manual_seed(123)  # the global RNG must not matter
    b = sampler((5,), generator=torch.Generator().manual_seed(0))
    torch.testing.assert_close(a, b)
    assert not torch.equal(a, sampler((5,), generator=torch.Generator().manual_seed(1)))


# -------------------- AbstractFlowMethodConfig --------------------
def test_family_base_config_is_not_buildable():
    with pytest.raises(pydantic.ValidationError, match="family base"):
        AbstractFlowMethodConfig()


def test_config_rejects_non_callable_sampler():
    with pytest.raises(pydantic.ValidationError, match="callable"):
        CFMTrainingConfig(time_sampler=3)


def test_config_with_live_sampler_is_not_portable():
    config = CFMTrainingConfig(noise_sampler=my_sampler)
    assert config.noise_sampler is my_sampler
    with pytest.raises(PortabilityError):
        config.to_spec()


def test_cfm_config_spec_round_trips():
    config = CFMTrainingConfig()
    assert CFMTrainingConfig.from_spec(config.to_spec()) == config


# -------------------- Protocols --------------------
def test_structural_protocols(module):
    assert isinstance(Trainer(module), SupportsTraining)
    assert not isinstance(Trainer(module), SupportsInference)
    assert isinstance(Predictor(module), SupportsInference)
    assert not isinstance(Predictor(module), SupportsTraining)


# -------------------- Matcher --------------------
def test_base_matcher_is_abstract():
    with pytest.raises(TypeError):
        BaseMatcher(match_fn=lambda **_: (None, None))


def test_match_unchanged_without_source_coupling():
    def match_fn(**_):
        raise AssertionError("match_fn must not be called")

    data = new_step_data(target_coupling_lin=torch.zeros(3, 2))
    assert Matcher(match_fn).match(data, rng=np.random.default_rng(0)) is data


def test_match_unchanged_when_indices_none(coupled):
    assert Matcher(lambda **_: (None, None)).match(coupled, rng=np.random.default_rng(0)) is coupled


def test_match_subscripts_and_passes_rng(coupled):
    rng = np.random.default_rng(0)
    seen = {}

    def match_fn(source_lin, target_lin, source_quad, target_quad, rng):
        seen.update(source_lin=source_lin, target_lin=target_lin, rng=rng)
        return torch.tensor([2, 0]), torch.tensor([1, 1])

    out = Matcher(match_fn).match(coupled, rng=rng)
    assert seen["rng"] is rng
    assert seen["source_lin"] is coupled["source_coupling_lin"]
    assert seen["target_lin"] is coupled["target_coupling_lin"]
    torch.testing.assert_close(out["source_state"], torch.tensor([[2.0], [0.0]]))
    torch.testing.assert_close(out["target_state"], torch.tensor([[4.0], [4.0]]))


# -------------------- MatchedTrainingMethod --------------------
def test_matched_training_method(module, coupled):
    inner = Trainer(module)
    matcher = Matcher(lambda **_: (torch.tensor([1]), torch.tensor([2])))
    matched = MatchedTrainingMethod(inner, matcher)
    generator, rng = torch.Generator().manual_seed(0), np.random.default_rng(0)

    loss, _ = matched.compute_loss(coupled, generator=generator, rng=rng)

    assert loss.item() == 1.0
    ((data, seen_generator, seen_rng),) = inner.calls
    assert seen_generator is generator
    assert seen_rng is rng
    torch.testing.assert_close(data["source_state"], torch.tensor([[1.0]]))
    torch.testing.assert_close(data["target_state"], torch.tensor([[5.0]]))
    assert matched.method is inner
    assert matched.matcher is matcher
    assert matched.module is module
    assert isinstance(matched, SupportsTraining)
