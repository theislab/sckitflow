from typing import Literal

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
    ProtocolMixin,
    ProtocolSpecs,
    TrainingProtocolWrapper,
)
from sckitflow.core.methods._opt import OptimConfig, OptimizationManager
from sckitflow.core.methods.inference._ode import ODEInference
from sckitflow.core.methods.training._cfm import CFMTrainingProtocol

TRAINING_PROTOCOLS_REGISTRY = {
    "cfm": CFMTrainingProtocol,
}
INFERENCE_PROTOCOLS_REGISTRY = {"ode": ODEInference}

AVAILABLE_TRAINING_PROTOCOLS = Literal["cfm"]
AVAILABLE_INFERENCE_PROTOCOLS = Literal["ode"]

__all__ = [
    "ProtocolSpecs",
    "FlowSpecs",
    "BaseTrainingProtocol",
    "BaseFlowTrainingProtocol",
    "BaseInferenceProtocol",
    "BaseFlowInferenceProtocol",
    "BaseMatchingProtocol",
    "MatchingProtocol",
    "ProtocolMixin",
    "TrainingProtocolWrapper",
    "InferenceProtocolWrapper",
    "MatchedTrainingProtocol",
    "CFM",
    "OptimConfig",
    "OptimizationManager",
    "TRAINING_PROTOCOLS_REGISTRY",
    "INFERENCE_PROTOCOLS_REGISTRY",
    "AVAILABLE_TRAINING_PROTOCOLS",
    "AVAILABLE_INFERENCE_PROTOCOLS",
]
