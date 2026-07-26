"""Back-compat shim: the component registry moved to :mod:`scfit.registry`.

The uniform :class:`~scfit.registry.Component` base + portable-spec (de)serialization now live in scfit, the
family-neutral core. New code should ``from scfit.registry import ...`` directly; this module re-exports the
public API so existing ``from sc_flow._registry import ...`` call sites keep working while the split lands.
"""

from __future__ import annotations

from scfit.registry import (
    Component,
    PortabilityError,
    build,
    parse,
    register_live,
    to_spec,
)

__all__ = ["Component", "PortabilityError", "build", "parse", "register_live", "to_spec"]
