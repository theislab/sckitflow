from typing import Any
from unittest.mock import patch

import pytest
import torch

from sckitflow.core._types import PredictionData, StepData
from sckitflow.core.methods._base import (
    BaseFlowInferenceProtocol,
    BaseFlowTrainingProtocol,
    BaseInferenceProtocol,
    BaseMatchingProtocol,
    BaseTrainingProtocol,
    FlowSpecs,
    InferenceProtocolWrapper,
    MatchedTrainingProtocol,
    MatchingProtocol,
    ProtocolSpecs,
    TrainingProtocolWrapper,
    _AbstractInferenceProtocol,
    _AbstractMatchingProtocol,
    _AbstractTrainingProtocol,
)


# -------------------- Dummy Implementations --------------------
class DummyModule(torch.nn.Module):
    """Simple module for testing; satisfies BaseModule informally."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 2)

    def forward(self, x):
        return self.linear(x)


class DummyStepData(dict):
    """Minimal StepData stand-in; a dict so subscripting works."""


class DummyPredictionData:
    """Minimal PredictionData stand-in."""


class DummyProbabilityPath:
    """Stand-in for BaseProbabilityPath; FlowSpecs only stores it."""


def dummy_time_sampler(shape, device=None, dtype=None):
    return torch.rand(shape, device=device, dtype=dtype)


def dummy_noise_sampler(shape, device=None, dtype=None):
    return torch.randn(shape, device=device, dtype=dtype)


# -------------------- Concrete Subclasses for Testing --------------------
class ConcreteTrainingProtocol(BaseTrainingProtocol):
    def compute_loss(self, step_data: StepData) -> tuple[torch.Tensor, dict[str, Any]]:
        return torch.tensor(0.0), {"loss": 0.0}


class ConcreteInferenceProtocol(BaseInferenceProtocol):
    def predict(self, step_data: StepData) -> PredictionData:
        return DummyPredictionData()


class ConcreteFlowTrainingProtocol(BaseFlowTrainingProtocol):
    def compute_loss(self, step_data: StepData) -> tuple[torch.Tensor, dict[str, Any]]:
        return torch.tensor(0.0), {"loss": 0.0}


class ConcreteFlowInferenceProtocol(BaseFlowInferenceProtocol):
    def predict(self, step_data: StepData) -> PredictionData:
        return DummyPredictionData()


# -------------------- Fixtures --------------------
@pytest.fixture
def dummy_module():
    return DummyModule()


@pytest.fixture
def flow_specs_kwargs():
    """Keyword args required by FlowSpecs beyond ``module``."""
    return {
        "probability_path": DummyProbabilityPath(),
        "time_sampler": dummy_time_sampler,
    }


@pytest.fixture
def step_data():
    return DummyStepData()


@pytest.fixture
def coupling_step_data():
    """StepData with all four coupling fields populated."""
    return DummyStepData(
        source_coupling_lin=torch.randn(2, 3),
        source_coupling_quad=torch.randn(2, 3, 3),
        target_coupling_lin=torch.randn(2, 3),
        target_coupling_quad=torch.randn(2, 3, 3),
    )


# -------------------- Abstractness Tests --------------------
@pytest.mark.parametrize(
    "cls",
    [
        _AbstractTrainingProtocol,
        _AbstractInferenceProtocol,
        _AbstractMatchingProtocol,
    ],
)
def test_abstract_contracts_cannot_be_instantiated(cls):
    with pytest.raises(TypeError):
        cls()


@pytest.mark.parametrize(
    "cls",
    [
        BaseTrainingProtocol,
        BaseInferenceProtocol,
    ],
)
def test_base_protocols_cannot_be_instantiated(cls, dummy_module):
    with pytest.raises(TypeError):
        cls(dummy_module)


@pytest.mark.parametrize(
    "cls",
    [
        BaseFlowTrainingProtocol,
        BaseFlowInferenceProtocol,
    ],
)
def test_base_flow_protocols_cannot_be_instantiated(cls, dummy_module, flow_specs_kwargs):
    with pytest.raises(TypeError):
        cls(dummy_module, **flow_specs_kwargs)


def test_base_matching_protocol_is_abstract():
    with pytest.raises(TypeError):
        BaseMatchingProtocol(match_fn=lambda **_: (None, None))


# -------------------- ProtocolSpecs --------------------
def test_protocol_specs_initialization_and_properties(dummy_module):
    proto = ProtocolSpecs(dummy_module, dtype=torch.float64, device_id="cpu")
    assert proto.dtype == torch.float64
    assert proto.device_id == "cpu"
    assert proto.module is dummy_module
    assert next(proto.module.parameters()).dtype == torch.float64


def test_protocol_specs_set_train_mode(dummy_module):
    proto = ProtocolSpecs(dummy_module)
    proto.set_train_mode(True)
    assert proto.module.training is True
    proto.set_train_mode(False)
    assert proto.module.training is False


# -------------------- FlowSpecs --------------------
def test_flow_specs_initialization_and_properties(dummy_module, flow_specs_kwargs):
    specs = FlowSpecs(
        dummy_module,
        dtype=torch.float64,
        device_id="cpu",
        **flow_specs_kwargs,
    )

    assert specs.module is dummy_module
    assert specs.dtype == torch.float64
    assert specs.device_id == "cpu"
    assert specs.probability_path is flow_specs_kwargs["probability_path"]
    assert specs.time_sampler is flow_specs_kwargs["time_sampler"]
    assert specs.noise_sampler is torch.randn
    assert specs.generate_from_noise is False


def test_flow_specs_with_noise_sampler(dummy_module, flow_specs_kwargs):
    specs = FlowSpecs(
        dummy_module,
        noise_sampler=dummy_noise_sampler,
        generate_from_noise=True,
        device_id="cpu",
        **flow_specs_kwargs,
    )
    assert specs.noise_sampler is dummy_noise_sampler
    assert specs.generate_from_noise is True


def test_flow_specs_generate_from_noise_requires_noise_sampler(dummy_module, flow_specs_kwargs):
    with pytest.raises(TypeError):
        FlowSpecs(
            dummy_module,
            generate_from_noise=True,
            noise_sampler=None,
            **flow_specs_kwargs,
        )


def test_flow_specs_set_train_mode(dummy_module, flow_specs_kwargs):
    specs = FlowSpecs(dummy_module, device_id="cpu", **flow_specs_kwargs)
    specs.set_train_mode(True)
    assert specs.module.training is True
    specs.set_train_mode(False)
    assert specs.module.training is False


def test_flow_specs_is_protocol_specs(dummy_module, flow_specs_kwargs):
    specs = FlowSpecs(dummy_module, device_id="cpu", **flow_specs_kwargs)
    assert isinstance(specs, ProtocolSpecs)


# -------------------- Concrete Protocol Subclasses --------------------
def test_training_protocol_subclass(dummy_module, step_data):
    proto = ConcreteTrainingProtocol(dummy_module, device_id="cpu")
    assert isinstance(proto, _AbstractTrainingProtocol)
    assert proto.dtype == torch.float32
    assert proto.device_id == "cpu"
    loss, meta = proto.compute_loss(step_data)
    assert loss.item() == 0.0
    assert meta == {"loss": 0.0}


def test_inference_protocol_subclass(dummy_module, step_data):
    proto = ConcreteInferenceProtocol(dummy_module, device_id="cpu")
    assert isinstance(proto, _AbstractInferenceProtocol)
    pred = proto.predict(step_data)
    assert isinstance(pred, DummyPredictionData)


def test_flow_training_protocol_subclass(dummy_module, step_data, flow_specs_kwargs):
    proto = ConcreteFlowTrainingProtocol(dummy_module, device_id="cpu", **flow_specs_kwargs)
    assert isinstance(proto, _AbstractTrainingProtocol)
    assert isinstance(proto, FlowSpecs)
    loss, _ = proto.compute_loss(step_data)
    assert loss.item() == 0.0


def test_flow_inference_protocol_subclass(dummy_module, step_data, flow_specs_kwargs):
    proto = ConcreteFlowInferenceProtocol(dummy_module, device_id="cpu", **flow_specs_kwargs)
    assert isinstance(proto, _AbstractInferenceProtocol)
    assert isinstance(proto, FlowSpecs)
    assert isinstance(proto.predict(step_data), DummyPredictionData)


# -------------------- Wrapper Delegation --------------------
def test_training_protocol_wrapper_delegates(dummy_module, step_data):
    inner = ConcreteTrainingProtocol(dummy_module, dtype=torch.float64, device_id="cpu")
    wrapper = TrainingProtocolWrapper(inner)

    loss, meta = wrapper.compute_loss(step_data)
    assert loss.item() == 0.0
    assert meta == {"loss": 0.0}

    assert wrapper.dtype == torch.float64
    assert wrapper.device_id == "cpu"
    assert wrapper.module is dummy_module
    assert wrapper.protocol is inner


def test_training_wrapper_accepts_flow_protocol(dummy_module, step_data, flow_specs_kwargs):
    inner = ConcreteFlowTrainingProtocol(dummy_module, dtype=torch.float64, device_id="cpu", **flow_specs_kwargs)
    wrapper = TrainingProtocolWrapper(inner)

    loss, _ = wrapper.compute_loss(step_data)
    assert loss.item() == 0.0
    assert wrapper.dtype == torch.float64
    assert wrapper.device_id == "cpu"
    assert wrapper.module is dummy_module
    assert wrapper.protocol is inner


def test_inference_protocol_wrapper_delegates(dummy_module, step_data):
    inner = ConcreteInferenceProtocol(dummy_module, dtype=torch.float64, device_id="cpu")
    wrapper = InferenceProtocolWrapper(inner)

    assert isinstance(wrapper.predict(step_data), DummyPredictionData)
    assert wrapper.dtype == torch.float64
    assert wrapper.device_id == "cpu"
    assert wrapper.module is dummy_module
    assert wrapper.protocol is inner


def test_inference_wrapper_accepts_flow_protocol(dummy_module, step_data, flow_specs_kwargs):
    inner = ConcreteFlowInferenceProtocol(dummy_module, dtype=torch.float64, device_id="cpu", **flow_specs_kwargs)
    wrapper = InferenceProtocolWrapper(inner)

    assert isinstance(wrapper.predict(step_data), DummyPredictionData)
    assert wrapper.protocol is inner


def test_wrapper_delegates_set_train_mode(dummy_module):
    inner = ConcreteTrainingProtocol(dummy_module)
    wrapper = TrainingProtocolWrapper(inner)

    wrapper.set_train_mode(True)
    assert inner.module.training is True
    wrapper.set_train_mode(False)
    assert inner.module.training is False


# -------------------- Wrapper Type Checks --------------------
def test_training_wrapper_rejects_non_training_protocol(dummy_module):
    # ConcreteInferenceProtocol is storage-backed but not a training protocol.
    with pytest.raises(TypeError):
        TrainingProtocolWrapper(ConcreteInferenceProtocol(dummy_module))


def test_inference_wrapper_rejects_non_inference_protocol(dummy_module):
    with pytest.raises(TypeError):
        InferenceProtocolWrapper(ConcreteTrainingProtocol(dummy_module))


# -------------------- MatchingProtocol --------------------
def test_matching_protocol_returns_unchanged_when_no_source(coupling_step_data):
    coupling_step_data["source_coupling_lin"] = None
    coupling_step_data["source_coupling_quad"] = None

    matcher = MatchingProtocol(match_fn=lambda **_: (None, None))
    result = matcher.match(coupling_step_data)

    assert result is coupling_step_data


def test_matching_protocol_returns_unchanged_when_indices_none(coupling_step_data):
    matcher = MatchingProtocol(match_fn=lambda **_: (None, None))
    result = matcher.match(coupling_step_data)

    assert result is coupling_step_data


def test_matching_protocol_calls_match_fn_with_all_fields(coupling_step_data):
    captured = {}

    def match_fn(source_lin, target_lin, source_quad, target_quad):
        captured.update(
            source_lin=source_lin,
            target_lin=target_lin,
            source_quad=source_quad,
            target_quad=target_quad,
        )
        return torch.tensor([0]), torch.tensor([0])

    matcher = MatchingProtocol(match_fn=match_fn)

    with patch("sckitflow.core.methods._base.subscript_step_data") as mock_sub:
        mock_sub.return_value = {"matched": True}
        matcher.match(coupling_step_data)

    assert captured["source_lin"] is coupling_step_data["source_coupling_lin"]
    assert captured["source_quad"] is coupling_step_data["source_coupling_quad"]
    assert captured["target_lin"] is coupling_step_data["target_coupling_lin"]
    assert captured["target_quad"] is coupling_step_data["target_coupling_quad"]


def test_matching_protocol_subscripts_when_indices_present(coupling_step_data):
    src_idxs = torch.tensor([0, 1])
    tgt_idxs = torch.tensor([1, 0])
    matcher = MatchingProtocol(match_fn=lambda **_: (src_idxs, tgt_idxs))

    with patch("sckitflow.core.methods._base.subscript_step_data") as mock_sub:
        mock_sub.return_value = {"matched": True}
        result = matcher.match(coupling_step_data)

    mock_sub.assert_called_once_with(coupling_step_data, src_idxs=src_idxs, tgt_idxs=tgt_idxs)
    assert result == {"matched": True}


# -------------------- MatchedTrainingProtocol --------------------
def test_matched_training_protocol_runs_match_then_compute_loss(dummy_module):
    order = []

    def match_fn(source_lin, target_lin, source_quad, target_quad):
        order.append("match")
        return torch.tensor([0]), torch.tensor([0])

    class RecordingTrainingProtocol(BaseTrainingProtocol):
        def compute_loss(self, step_data):
            order.append(("compute_loss", step_data))
            return torch.tensor(1.0), {}

    inner = RecordingTrainingProtocol(dummy_module)
    matched = MatchedTrainingProtocol(inner, match_fn=match_fn)

    step_data = DummyStepData(
        source_coupling_lin=torch.randn(2, 3),
        source_coupling_quad=torch.randn(2, 3, 3),
        target_coupling_lin=torch.randn(2, 3),
        target_coupling_quad=torch.randn(2, 3, 3),
    )

    matched_data = {"matched": True}
    with patch("sckitflow.core.methods._base.subscript_step_data") as mock_sub:
        mock_sub.return_value = matched_data
        loss, _ = matched.compute_loss(step_data)

    assert loss.item() == 1.0
    assert order[0] == "match"
    assert order[1] == ("compute_loss", matched_data)


def test_matched_training_protocol_accepts_flow_protocol(dummy_module, flow_specs_kwargs):
    inner = ConcreteFlowTrainingProtocol(dummy_module, device_id="cpu", **flow_specs_kwargs)
    matched = MatchedTrainingProtocol(inner, match_fn=lambda **_: (None, None))
    assert isinstance(matched.matcher, MatchingProtocol)


def test_matched_training_protocol_exposes_matcher(dummy_module):
    inner = ConcreteTrainingProtocol(dummy_module)
    matched = MatchedTrainingProtocol(inner, match_fn=lambda **_: (None, None))

    assert isinstance(matched.matcher, MatchingProtocol)


def test_matched_training_protocol_skips_match_when_no_source(dummy_module):
    def match_fn(**kwargs):
        raise AssertionError("match_fn should not be called when no source")

    inner = ConcreteTrainingProtocol(dummy_module)
    matched = MatchedTrainingProtocol(inner, match_fn=match_fn)

    step_data = DummyStepData(
        source_coupling_lin=None,
        source_coupling_quad=None,
        target_coupling_lin=torch.randn(2, 3),
        target_coupling_quad=torch.randn(2, 3, 3),
    )

    loss, _ = matched.compute_loss(step_data)
    assert loss.item() == 0.0
