"""Back-compat re-export.

``StepData`` moved to :mod:`sc_flow.backends.torch._types` (its canonical home,
shared with non-training consumers like the inverse-problem surrogate wrappers).
Import it from there; this shim keeps existing ``methods._utils`` imports working.
"""

from sc_flow.backends.torch._types import StepData

__all__ = ["StepData"]
