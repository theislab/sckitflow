import functools
import inspect
from typing import Any

from jaxtyping import PyTree

from sc_flow.backends.jax._types import TDiffusion, TTimeStateDiffusion


def _prepare_diffusion_fn(diffusion_fn: TDiffusion, df_kwargs: PyTree[Any] | None = None) -> TTimeStateDiffusion:
    df_kwargs = df_kwargs or {}
    partial_diffusion = functools.partial(diffusion_fn, **df_kwargs)

    sig = inspect.signature(diffusion_fn)
    params = sig.parameters

    remaining_params = [p for p in params if p not in df_kwargs]
    num_params = len(remaining_params)

    return partial_diffusion, num_params
