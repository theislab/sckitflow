from typing import Any

from torchax.interop import JittableModule

from sc_flow.backends.torch._types import MappedTensor, TVfFn
from sc_flow.backends.torch.nn._vf import BaseVelocityField
from sc_flow.backends.torch.nn._vf import MLPUnconditionalVF as _TorchMLPUnconditionalVF

__all__ = ["BaseVelocityField", "MLPUnconditionalVF"]


class MLPUnconditionalVF(_TorchMLPUnconditionalVF):
    """Torchax unconditional velocity field."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.to("jax")

    def get_vf_fn(
        self,
        condition_dict: MappedTensor | None = None,
        source: Any | None = None,
    ) -> TVfFn:
        """Return a JIT-compiled velocity field function for use in solvers.

        The returned callable has signature ``(t, x) -> v`` and is compiled by
        XLA on the first invocation.  Subsequent calls with the same input shapes
        hit the cache and incur no recompilation cost.
        """
        jitted = JittableModule(self)

        def _vf_fn(t, x):
            return jitted(t, x, condition_dict=condition_dict, source=source)

        return _vf_fn
