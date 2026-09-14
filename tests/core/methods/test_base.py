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
    SupportsInference,
    SupportsProtocol,
    SupportsTraining,
    TrainingProtocolWrapper,
    _AbstractInferenceProtocol,
    _AbstractMatchingProtocol,
    _AbstractTrainingProtocol,
)


# -----------------------------------------------------------------------------
# Dummies
# -----------------------------------------------------------------------------
class DummyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x):
        return self.linear(x)


class DummyStepData(dict):
    """Minimal `StepData` stand-in.

    Always carries the four coupling keys (defaulting to `None`), matching the
    contract `MatchingProtocol.match` relies on.
    """

    def __init__(self, **kwargs):
        super().__init__(
            source_coupling_lin=None,
            source_coupling_quad=None,
            target_coupling_lin=None,
            target_coupling_quad=None,
        )
        self.update(kwargs)


class DummyPredictionData:
    def __init__(self, X=None):
        self.X = X if X is not None else torch.zeros(1, 4)
        self.traj = None
        self.raw_samples = None


def dummy_time_sampler(shape, device=None, dtype=None):
    return torch.rand(shape, device=device, dtype=dtype)


def dummy_noise_sampler(shape, device=None, dtype=None):
    return torch.randn(shape, device=device, dtype=dtype)


# -----------------------------------------------------------------------------
# Concrete protocols
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture
def dummy_module():
    return DummyModule()


@pytest.fixture
def flow_kwargs():
    return {"time_sampler": dummy_time_sampler}


@pytest.fixture
def coupling_step_data():
    return DummyStepData(
        source_coupling_lin=torch.randn(2, 3),
        source_coupling_quad=torch.randn(2, 3, 3),
        target_coupling_lin=torch.randn(2, 3),
        target_coupling_quad=torch.randn(2, 3, 3),
    )


# -----------------------------------------------------------------------------
# Abstract contracts
# -----------------------------------------------------------------------------
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
    [BaseTrainingProtocol, BaseInferenceProtocol],
)
def test_base_protocols_cannot_be_instantiated(cls, dummy_module):
    with pytest.raises(TypeError):
        cls(dummy_module)


@pytest.mark.parametrize(
    "cls",
    [BaseFlowTrainingProtocol, BaseFlowInferenceProtocol],
)
def test_base_flow_protocols_cannot_be_instantiated(cls, dummy_module, flow_kwargs):
    with pytest.raises(TypeError):
        cls(dummy_module, **flow_kwargs)


def test_base_matching_protocol_is_abstract():
    with pytest.raises(TypeError):
        BaseMatchingProtocol(match_fn=lambda **_: (None, None))


# -----------------------------------------------------------------------------
# ProtocolSpecs / FlowSpecs
# -----------------------------------------------------------------------------
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


def test_flow_specs_defaults(dummy_module):
    """`FlowSpecs` auto-fills probability path, time and noise samplers."""
    specs = FlowSpecs(dummy_module, device_id="cpu")
    assert specs.probability_path is not None
    assert specs.time_sampler is torch.rand
    assert specs.noise_sampler is torch.randn
    assert specs.generate_from_noise is False


def test_flow_specs_custom_values(dummy_module, flow_kwargs):
    specs = FlowSpecs(
        dummy_module,
        noise_sampler=dummy_noise_sampler,
        generate_from_noise=True,
        device_id="cpu",
        **flow_kwargs,
    )
    assert specs.time_sampler is dummy_time_sampler
    assert specs.noise_sampler is dummy_noise_sampler
    assert specs.generate_from_noise is True


def test_flow_specs_is_protocol_specs(dummy_module):
    specs = FlowSpecs(dummy_module, device_id="cpu")
    assert isinstance(specs, ProtocolSpecs)


# -----------------------------------------------------------------------------
# Structural contracts
# -----------------------------------------------------------------------------
def test_protocol_specs_satisfies_storage_protocol(dummy_module):
    assert isinstance(ProtocolSpecs(dummy_module), SupportsProtocol)


def test_flow_specs_satisfies_storage_protocol(dummy_module):
    assert isinstance(FlowSpecs(dummy_module, device_id="cpu"), SupportsProtocol)


def test_concrete_training_protocol_satisfies_supports_training(dummy_module):
    specs = ProtocolSpecs(dummy_module, device_id="cpu")
    proto = ConcreteTrainingProtocol(specs)
    assert isinstance(proto, SupportsTraining)
    assert isinstance(proto, SupportsProtocol)


def test_concrete_flow_training_protocol_satisfies_supports_training(dummy_module, flow_kwargs):
    specs = FlowSpecs(dummy_module, device_id="cpu", **flow_kwargs)
    proto = ConcreteFlowTrainingProtocol(specs)
    assert isinstance(proto, SupportsTraining)


def test_concrete_inference_protocol_satisfies_supports_inference(dummy_module):
    specs = ProtocolSpecs(dummy_module, device_id="cpu")
    proto = ConcreteInferenceProtocol(specs)
    assert isinstance(proto, SupportsInference)
    assert isinstance(proto, SupportsProtocol)


def test_concrete_flow_inference_protocol_satisfies_supports_inference(dummy_module, flow_kwargs):
    specs = FlowSpecs(dummy_module, device_id="cpu", **flow_kwargs)
    proto = ConcreteFlowInferenceProtocol(specs)
    assert isinstance(proto, SupportsInference)


def test_matched_training_protocol_satisfies_supports_training(dummy_module):
    """The whole point: the wrapper itself satisfies `SupportsTraining`."""
    specs = ProtocolSpecs(dummy_module, device_id="cpu")
    inner = ConcreteTrainingProtocol(specs)
    matched = MatchedTrainingProtocol(inner, match_fn=lambda **_: (None, None))
    assert isinstance(matched, SupportsTraining)


def test_training_wrapper_satisfies_supports_training(dummy_module):
    specs = ProtocolSpecs(dummy_module, device_id="cpu")
    inner = ConcreteTrainingProtocol(specs)
    wrapped = TrainingProtocolWrapper(inner)
    assert isinstance(wrapped, SupportsTraining)


def test_inference_wrapper_satisfies_supports_inference(dummy_module):
    specs = ProtocolSpecs(dummy_module, device_id="cpu")
    inner = ConcreteInferenceProtocol(specs)
    wrapped = InferenceProtocolWrapper(inner)
    assert isinstance(wrapped, SupportsInference)


def test_training_protocol_does_not_satisfy_supports_inference(dummy_module):
    """A training protocol has no `predict`."""
    specs = ProtocolSpecs(dummy_module, device_id="cpu")
    proto = ConcreteTrainingProtocol(specs)
    assert not isinstance(proto, SupportsInference)


def test_inference_protocol_does_not_satisfy_supports_training(dummy_module):
    """An inference protocol has no `compute_loss`."""
    specs = ProtocolSpecs(dummy_module, device_id="cpu")
    proto = ConcreteInferenceProtocol(specs)
    assert not isinstance(proto, SupportsTraining)


# -----------------------------------------------------------------------------
# Concrete protocol subclasses
# -----------------------------------------------------------------------------
def test_training_protocol_subclass(dummy_module):
    specs = ProtocolSpecs(dummy_module, device_id="cpu")
    proto = ConcreteTrainingProtocol(specs)
    assert isinstance(proto, _AbstractTrainingProtocol)
    assert proto.dtype == torch.float32
    assert proto.device_id == "cpu"
    loss, meta = proto.compute_loss(DummyStepData())
    assert loss.item() == 0.0
    assert meta == {"loss": 0.0}


def test_inference_protocol_subclass(dummy_module):
    specs = ProtocolSpecs(dummy_module, device_id="cpu")
    proto = ConcreteInferenceProtocol(specs)
    assert isinstance(proto, _AbstractInferenceProtocol)
    assert isinstance(proto.predict(DummyStepData()), DummyPredictionData)


def test_flow_training_protocol_subclass(dummy_module, flow_kwargs):
    specs = FlowSpecs(dummy_module, device_id="cpu", **flow_kwargs)
    proto = ConcreteFlowTrainingProtocol(specs)
    assert isinstance(proto, _AbstractTrainingProtocol)
    assert isinstance(specs, FlowSpecs)
    loss, _ = proto.compute_loss(DummyStepData())
    assert loss.item() == 0.0


def test_flow_inference_protocol_subclass(dummy_module, flow_kwargs):
    specs = FlowSpecs(dummy_module, device_id="cpu", **flow_kwargs)
    proto = ConcreteFlowInferenceProtocol(specs)
    assert isinstance(proto, _AbstractInferenceProtocol)
    assert isinstance(specs, FlowSpecs)
    assert isinstance(proto.predict(DummyStepData()), DummyPredictionData)


# -----------------------------------------------------------------------------
# TrainingProtocolWrapper
# -----------------------------------------------------------------------------
def test_training_wrapper_delegates_compute_loss(dummy_module):
    specs = ProtocolSpecs(dummy_module, dtype=torch.float64, device_id="cpu")
    inner = ConcreteTrainingProtocol(specs)
    wrapper = TrainingProtocolWrapper(inner)

    loss, meta = wrapper.compute_loss(DummyStepData())
    assert loss.item() == 0.0
    assert meta == {"loss": 0.0}

    assert wrapper.dtype == torch.float64
    assert wrapper.device_id == "cpu"
    assert wrapper.module is dummy_module
    assert wrapper.protocol is inner


def test_training_wrapper_accepts_flow_protocol(dummy_module, flow_kwargs):
    specs = FlowSpecs(dummy_module, device_id="cpu", **flow_kwargs)
    inner = ConcreteFlowTrainingProtocol(specs)
    wrapper = TrainingProtocolWrapper(inner)
    loss, _ = wrapper.compute_loss(DummyStepData())
    assert loss.item() == 0.0
    assert wrapper.protocol is inner


def test_training_wrapper_accepts_matched_protocol(dummy_module):
    """The wrapper accepts anything `SupportsTraining`, including another wrapper."""
    specs = ProtocolSpecs(dummy_module, device_id="cpu")
    inner = ConcreteTrainingProtocol(specs)
    matched = MatchedTrainingProtocol(inner, match_fn=lambda **_: (None, None))
    wrapper = TrainingProtocolWrapper(matched)
    loss, _ = wrapper.compute_loss(DummyStepData())
    assert loss.item() == 0.0
    assert wrapper.protocol is matched


def test_training_wrapper_rejects_inference_protocol(dummy_module):
    """An inference protocol does not satisfy `SupportsTraining`."""
    specs = ProtocolSpecs(dummy_module, device_id="cpu")
    inner = ConcreteInferenceProtocol(specs)
    with pytest.raises(TypeError, match="compute_loss"):
        TrainingProtocolWrapper(inner)


def test_training_wrapper_rejects_bare_specs(dummy_module):
    """A bare `ProtocolSpecs` has no `compute_loss`."""
    specs = ProtocolSpecs(dummy_module, device_id="cpu")
    with pytest.raises(TypeError, match="compute_loss"):
        TrainingProtocolWrapper(specs)


# -----------------------------------------------------------------------------
# InferenceProtocolWrapper
# -----------------------------------------------------------------------------
def test_inference_wrapper_delegates_predict(dummy_module):
    specs = ProtocolSpecs(dummy_module, dtype=torch.float64, device_id="cpu")
    inner = ConcreteInferenceProtocol(specs)
    wrapper = InferenceProtocolWrapper(inner)

    assert isinstance(wrapper.predict(DummyStepData()), DummyPredictionData)
    assert wrapper.dtype == torch.float64
    assert wrapper.device_id == "cpu"
    assert wrapper.module is dummy_module
    assert wrapper.protocol is inner


def test_inference_wrapper_accepts_flow_protocol(dummy_module, flow_kwargs):
    specs = FlowSpecs(dummy_module, device_id="cpu", **flow_kwargs)
    inner = ConcreteFlowInferenceProtocol(specs)
    wrapper = InferenceProtocolWrapper(inner)
    assert isinstance(wrapper.predict(DummyStepData()), DummyPredictionData)
    assert wrapper.protocol is inner


def test_inference_wrapper_rejects_training_protocol(dummy_module):
    specs = ProtocolSpecs(dummy_module, device_id="cpu")
    inner = ConcreteTrainingProtocol(specs)
    with pytest.raises(TypeError, match="predict"):
        InferenceProtocolWrapper(inner)


def test_inference_wrapper_rejects_bare_specs(dummy_module):
    specs = ProtocolSpecs(dummy_module, device_id="cpu")
    with pytest.raises(TypeError, match="predict"):
        InferenceProtocolWrapper(specs)


# -----------------------------------------------------------------------------
# set_train_mode delegation
# -----------------------------------------------------------------------------
def test_wrapper_delegates_set_train_mode(dummy_module):
    specs = ProtocolSpecs(dummy_module)
    inner = ConcreteTrainingProtocol(specs)
    wrapper = TrainingProtocolWrapper(inner)

    wrapper.set_train_mode(True)
    assert inner.module.training is True
    wrapper.set_train_mode(False)
    assert inner.module.training is False


# -----------------------------------------------------------------------------
# MatchingProtocol
# -----------------------------------------------------------------------------
def test_matching_protocol_returns_unchanged_when_no_source(coupling_step_data):
    coupling_step_data["source_coupling_lin"] = None
    coupling_step_data["source_coupling_quad"] = None

    matcher = MatchingProtocol(match_fn=lambda **_: (None, None))
    assert matcher.match(coupling_step_data) is coupling_step_data


def test_matching_protocol_returns_unchanged_when_indices_none(coupling_step_data):
    matcher = MatchingProtocol(match_fn=lambda **_: (None, None))
    assert matcher.match(coupling_step_data) is coupling_step_data


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


# -----------------------------------------------------------------------------
# MatchedTrainingProtocol
# -----------------------------------------------------------------------------
def test_matched_training_protocol_runs_match_then_compute_loss(dummy_module):
    order = []

    def match_fn(source_lin, target_lin, source_quad, target_quad):
        order.append("match")
        return torch.tensor([0]), torch.tensor([0])

    class RecordingTrainingProtocol(BaseTrainingProtocol):
        def compute_loss(self, step_data):
            order.append(("compute_loss", step_data))
            return torch.tensor(1.0), {}

    specs = ProtocolSpecs(dummy_module)
    inner = RecordingTrainingProtocol(specs)
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


def test_matched_training_protocol_exposes_matcher(dummy_module):
    specs = ProtocolSpecs(dummy_module)
    inner = ConcreteTrainingProtocol(specs)
    matched = MatchedTrainingProtocol(inner, match_fn=lambda **_: (None, None))
    assert isinstance(matched.matcher, MatchingProtocol)


def test_matched_training_protocol_accepts_flow_protocol(dummy_module, flow_kwargs):
    specs = FlowSpecs(dummy_module, device_id="cpu", **flow_kwargs)
    inner = ConcreteFlowTrainingProtocol(specs)
    matched = MatchedTrainingProtocol(inner, match_fn=lambda **_: (None, None))
    assert matched.protocol is inner
    assert isinstance(matched.matcher, MatchingProtocol)


def test_matched_training_protocol_skips_match_when_no_source(dummy_module):
    def match_fn(**kwargs):
        raise AssertionError("match_fn should not be called when no source")

    specs = ProtocolSpecs(dummy_module)
    inner = ConcreteTrainingProtocol(specs)
    matched = MatchedTrainingProtocol(inner, match_fn=match_fn)

    step_data = DummyStepData(
        source_coupling_lin=None,
        source_coupling_quad=None,
        target_coupling_lin=torch.randn(2, 3),
        target_coupling_quad=torch.randn(2, 3, 3),
    )

    loss, _ = matched.compute_loss(step_data)
    assert loss.item() == 0.0
