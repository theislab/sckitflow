"""Machine-readable descriptor of what a method needs and supports.

The framework (builder / ``from_config`` / validator) reasons about a method
through its :class:`MethodCapabilities` *only*. This is the seam that keeps the
framework generic: it can validate device / backend / solver combinations
without knowing what a velocity field or an ODE solver is. Anything
method-specific lives behind ``config_cls``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["MethodCapabilities"]


@dataclass(frozen=True)
class MethodCapabilities:
    """Describes the requirements and supported settings of a method.

    :param category: Method category, e.g. ``"flow"`` (has continuous dynamics /
        a velocity field and needs a solver) or ``"general"``.
    :param backends: Backends the method can run on.
    :param supported_devices: Devices the method can run on. The validator
        rejects any ``trainer.device`` not in this set — this is how, e.g., a
        through-ODE torchax method excludes ``"mps"`` without the framework
        naming a solver.
    :param config_cls: The method-specific config dataclass that ``method.config``
        is validated against. ``None`` means the method takes free-form kwargs.
    """

    category: str = "general"
    backends: frozenset[str] = field(default_factory=lambda: frozenset({"torch"}))
    supported_devices: frozenset[str] = field(default_factory=lambda: frozenset({"cpu", "cuda", "mps"}))
    config_cls: type | None = None
