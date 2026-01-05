from collections.abc import Callable
from typing import Any

from sc_flow._runtime import attempt_numba_import

__all__ = ["maybe_jit_fn"]


def maybe_jit_fn(
    fn: Callable[[Any], Any] | None = None,
    *,
    jit_kwargs: dict[str, Any] | None = None,
) -> Callable[[Any], Any]:
    def decorator(func: Callable[[Any], Any]) -> Callable[[Any], Any]:
        nb = attempt_numba_import()
        if nb is not None:
            return nb.jit(func, **(jit_kwargs or {}))
        return func

    if fn is not None:
        return decorator(fn)

    return decorator
