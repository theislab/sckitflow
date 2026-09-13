from typing import Any
from unittest.mock import patch

import pytest
import torch

from sckitflow.core._types import PredictionData, StepData
from sckitflow.core.methods._base import (
    BaseInferenceProtocol,
    BaseMatchingProtocol,
    BaseMethod,
    BaseTrainingProtocol,
    FlowMethodSpecs,
    InferenceProtocolWrapper,
    MatchedProtocolSpecs,
    MatchedTrainingProtocol,
    MatchingProtocol,
    MatchingSpecs,
    MethodWrapper,
    ProtocolSpecs,
    TrainingProtocolWrapper,
    _AbstractInferenceProtocol,
    _AbstractMatchingProtocol,
    _AbstractMethod,
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


# -------------------- Concrete Subclasses for Testing --------------------
class ConcreteTrainingProtocol(BaseTrainingProtocol):
    def train_step(self, step_data: StepData) -> tuple[torch.Tensor, dict[str, Any]]:
        return torch.tensor(0.0), {"loss": 0.0}


class ConcreteInferenceProtocol(BaseInferenceProtocol):
    def predict(self, step_data: StepData) -> PredictionData:
        return DummyPredictionData()


class ConcreteMethod(BaseMethod):
    def train_step(self, step_data: StepData) -> tuple[torch.Tensor, dict[str, Any]]:
        return torch.tensor(0.0), {"loss": 0.0}

    def predict(self, step_data: StepData) -> PredictionData:
        return DummyPredictionData()


class ConcreteMatchingProtocol(BaseMatchingProtocol):
    def match(self, step_data: StepData) -> StepData:
        return step_data


# -------------------- Fixtures --------------------
@pytest.fixture
def dummy_module():
    return DummyModule()


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
    "cls, needs_module",
    [
        (_AbstractTrainingProtocol, False),
        (_AbstractInferenceProtocol, False),
        (_AbstractMatchingProtocol, False),
        (_AbstractMethod, False),
        (BaseTrainingProtocol, True),
        (BaseInferenceProtocol, True),
        (BaseMethod, True),
    ],
)
def test_abstract_classes_cannot_be_instantiated(cls, needs_module, dummy_module):
    with pytest.raises(TypeError):
        if needs_module:
            cls(dummy_module)
        else:
            cls()


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


# -------------------- MatchingSpecs --------------------
def test_matching_specs_stores_match_fn():
    fn = lambda **_: (None, None)
    specs = MatchingSpecs(fn)
    assert specs.match_fn is fn


def test_matching_specs_match_fn_is_read_only():
    specs = MatchingSpecs(lambda **_: (None, None))
    with pytest.raises(AttributeError):
        specs.match_fn = lambda **_: (None, None)


# -------------------- MatchedProtocolSpecs --------------------
def test_matched_protocol_specs_initializes_both_bases(dummy_module):
    """MatchedProtocolSpecs should initialize storage and matching state."""
    fn = lambda **_: (None, None)
    specs = MatchedProtocolSpecs(
        module=dummy_module,
        match_fn=fn,
        dtype=torch.float64,
        device_id="cpu",
    )

    # Storage side
    assert specs.module is dummy_module
    assert specs.dtype == torch.float64
    assert specs.device_id == "cpu"
    assert next(specs.module.parameters()).dtype == torch.float64

    # Matching side
    assert specs.match_fn is fn


def test_matched_protocol_specs_defaults(dummy_module):
    """Defaults should match ProtocolSpecs defaults and carry the matcher."""
    fn = lambda **_: (None, None)
    specs = MatchedProtocolSpecs(dummy_module, match_fn=fn)

    assert specs.dtype == torch.float32
    # device_id default depends on availability; just ensure it is a string
    assert isinstance(specs.device_id, str)


def test_matched_protocol_specs_set_train_mode(dummy_module):
    """set_train_mode from ProtocolSpecs should work through MatchedProtocolSpecs."""
    fn = lambda **_: (None, None)
    specs = MatchedProtocolSpecs(dummy_module, match_fn=fn)

    specs.set_train_mode(True)
    assert specs.module.training is True
    specs.set_train_mode(False)
    assert specs.module.training is False


def test_matched_protocol_specs_mro():
    """MatchedProtocolSpecs should be a ProtocolSpecs and a MatchingSpecs."""
    assert issubclass(MatchedProtocolSpecs, ProtocolSpecs)
    assert issubclass(MatchedProtocolSpecs, MatchingSpecs)


def test_matched_protocol_specs_usable_as_both_mixins(dummy_module):
    """A subclass combining MatchedProtocolSpecs with contracts should work."""
    fn = lambda **_: (None, None)

    class ConcreteMatchedTrainingProtocol(
        MatchedProtocolSpecs,
        _AbstractTrainingProtocol,
        _AbstractMatchingProtocol,
    ):
        def train_step(self, step_data):
            return torch.tensor(0.0), {}

        def match(self, step_data):
            return step_data

    proto = ConcreteMatchedTrainingProtocol(module=dummy_module, match_fn=fn, device_id="cpu")
    assert proto.dtype == torch.float32
    assert proto.device_id == "cpu"
    assert proto.match_fn is fn
    assert proto.match(DummyStepData()) is not None


def test_matched_protocol_specs_requires_module_and_match_fn(dummy_module):
    """Both required arguments must be supplied."""
    with pytest.raises(TypeError):
        MatchedProtocolSpecs(match_fn=lambda **_: (None, None))  # no module

    with pytest.raises(TypeError):
        MatchedProtocolSpecs(dummy_module)  # no match_fn


# -------------------- FlowMethodSpecs --------------------
def test_flow_method_specs_initializes_all_attributes(dummy_module):
    fn = lambda **_: (None, None)
    ns = lambda *a, **k: torch.randn(1)
    ts = lambda *a, **k: torch.rand(1)
    path = object()  # stand-in for BaseProbabilityPath

    specs = FlowMethodSpecs(
        module=dummy_module,
        match_fn=fn,
        dtype=torch.float64,
        device_id="cpu",
        probability_path=path,
        noise_sampler=ns,
        time_sampler=ts,
        generate_from_noise=True,
    )

    # Inherited from MatchedProtocolSpecs
    assert specs.module is dummy_module
    assert specs.dtype == torch.float64
    assert specs.device_id == "cpu"
    assert specs.match_fn is fn

    # Flow-specific
    assert specs.probability_path is path
    assert specs.noise_sampler is ns
    assert specs.time_sampler is ts
    assert specs.generate_from_noise is True


def test_flow_method_specs_defaults(dummy_module):
    fn = lambda **_: (None, None)
    specs = FlowMethodSpecs(dummy_module, match_fn=fn)

    assert specs.probability_path is None
    assert specs.noise_sampler is None
    assert specs.time_sampler is None
    assert specs.generate_from_noise is False


def test_flow_method_specs_requires_module_and_match_fn(dummy_module):
    with pytest.raises(TypeError):
        FlowMethodSpecs(match_fn=lambda **_: (None, None))  # no module

    with pytest.raises(TypeError):
        FlowMethodSpecs(dummy_module)  # no match_fn


def test_flow_method_specs_mro():
    """FlowMethodSpecs should be a MatchedProtocolSpecs (and transitively both mixins)."""
    assert issubclass(FlowMethodSpecs, MatchedProtocolSpecs)
    assert issubclass(FlowMethodSpecs, ProtocolSpecs)
    assert issubclass(FlowMethodSpecs, MatchingSpecs)


def test_flow_method_specs_set_train_mode(dummy_module):
    fn = lambda **_: (None, None)
    specs = FlowMethodSpecs(dummy_module, match_fn=fn)

    specs.set_train_mode(True)
    assert specs.module.training is True
    specs.set_train_mode(False)
    assert specs.module.training is False


def test_flow_method_specs_usable_as_mixin(dummy_module):
    """FlowMethodSpecs should compose with abstract contracts."""
    fn = lambda **_: (None, None)

    class ConcreteFlowMethod(
        FlowMethodSpecs,
        _AbstractTrainingProtocol,
        _AbstractInferenceProtocol,
        _AbstractMatchingProtocol,
    ):
        def train_step(self, step_data):
            return torch.tensor(0.0), {}

        def predict(self, step_data):
            return DummyPredictionData()

        def match(self, step_data):
            return step_data

    proto = ConcreteFlowMethod(
        module=dummy_module,
        match_fn=fn,
        device_id="cpu",
        generate_from_noise=True,
    )
    assert proto.match_fn is fn
    assert proto.generate_from_noise is True
    assert proto.probability_path is None
    assert isinstance(proto.predict(DummyStepData()), DummyPredictionData)
    loss, _ = proto.train_step(DummyStepData())
    assert loss.item() == 0.0


# -------------------- Concrete Protocol Subclasses --------------------
def test_training_protocol_subclass(dummy_module, step_data):
    proto = ConcreteTrainingProtocol(dummy_module, device_id="cpu")
    assert isinstance(proto, _AbstractTrainingProtocol)
    assert proto.dtype == torch.float32
    assert proto.device_id == "cpu"
    loss, meta = proto.train_step(step_data)
    assert loss.item() == 0.0
    assert meta == {"loss": 0.0}


def test_inference_protocol_subclass(dummy_module, step_data):
    proto = ConcreteInferenceProtocol(dummy_module, device_id="cpu")
    assert isinstance(proto, _AbstractInferenceProtocol)
    pred = proto.predict(step_data)
    assert isinstance(pred, DummyPredictionData)


def test_method_subclass(dummy_module, step_data):
    proto = ConcreteMethod(dummy_module, device_id="cpu")
    assert isinstance(proto, _AbstractMethod)
    loss, _ = proto.train_step(step_data)
    assert loss.item() == 0.0
    assert isinstance(proto.predict(step_data), DummyPredictionData)
    assert proto.dtype == torch.float32
    assert proto.device_id == "cpu"


# -------------------- Wrapper Delegation --------------------
def test_training_protocol_wrapper_delegates(dummy_module, step_data):
    inner = ConcreteTrainingProtocol(dummy_module, dtype=torch.float64, device_id="cpu")
    wrapper = TrainingProtocolWrapper(inner)

    loss, meta = wrapper.train_step(step_data)
    assert loss.item() == 0.0
    assert meta == {"loss": 0.0}

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


def test_method_wrapper_delegates_both(dummy_module, step_data):
    inner = ConcreteMethod(dummy_module, dtype=torch.float64, device_id="cpu")
    wrapper = MethodWrapper(inner)

    loss, _ = wrapper.train_step(step_data)
    assert loss.item() == 0.0
    assert isinstance(wrapper.predict(step_data), DummyPredictionData)
    assert wrapper.dtype == torch.float64
    assert wrapper.device_id == "cpu"
    assert wrapper.module is dummy_module
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


def test_method_wrapper_rejects_non_method_protocol(dummy_module):
    with pytest.raises(TypeError):
        MethodWrapper(ConcreteTrainingProtocol(dummy_module))


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
def test_matched_training_protocol_runs_match_then_train(dummy_module):
    order = []

    def match_fn(source_lin, target_lin, source_quad, target_quad):
        order.append("match")
        return torch.tensor([0]), torch.tensor([0])

    class RecordingTrainingProtocol(BaseTrainingProtocol):
        def train_step(self, step_data):
            order.append(("train", step_data))
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
        loss, _ = matched.train_step(step_data)

    assert loss.item() == 1.0
    assert order[0] == "match"
    assert order[1] == ("train", matched_data)


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

    loss, _ = matched.train_step(step_data)
    assert loss.item() == 0.0
