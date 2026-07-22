"""sc_flow.core — the reusable ML-toolbox base (separable from the flow-matching layer).

Holds what is generic across models: the ``data`` streaming/splitting layer (over ``binded``), the
PyTorch-Lightning training harness + optimizer, generic ``nn`` backbones, and evaluation ``metrics``.
The flow-matching specifics (velocity fields, probability paths, objectives, predict, OT coupling) live
in the sibling :mod:`sc_flow.flow`.
"""

from sc_flow.core._component import ComponentRegistry, ComponentSpec, JsonValue

__all__ = [
    "ComponentRegistry",
    "ComponentSpec",
    "JsonValue",
]
