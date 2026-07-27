"""Back-compat shim: the activation resolver moved to :mod:`scfit.nn`. Import from there in new code."""

from scfit.nn._activation import _ACTIVATIONS, ActivationId, resolve_activation

__all__ = ["ActivationId", "resolve_activation"]
