from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import PyTree
from torchax.interop import jax_view, torch_view
from torchax.tensor import Tensor as TorchaxTensor

from sc_flow.backends.jax.solvers._utils import _prepare_diffusion_fn
from sc_flow.backends.torch._types import TDiffusion, TODEDynamics, TSDEDynamics

__all__ = [
    "_get_drift_fn_wrapper",
    "_get_diffusion_fn_wrapper",
    "_init_wrapped_ode_dynamics",
    "_init_wrapped_sde_terms",
]


def _get_drift_fn_wrapper(
    dynamics: TODEDynamics,
    vf_kwargs: dict[str, Any],
) -> Callable:
    # prepare argument in higer scope
    vf_kwargs = vf_kwargs or {}

    def _call_wrapped_drift_fn(t, x, args=None):
        # handle backend
        t_torch = torch_view(t)
        x_torch = torch_view(x)

        # prepare arguments

        # call with extra arguments
        # copy attributes to ensure differentiablity
        if args is not None and "params" in args:
            params_jax = args["params"]
            param_names = args["param_names"]

            orig = {n: getattr(dynamics, n) for n in param_names}
            for n, v in zip(param_names, params_jax, strict=False):
                object.__setattr__(dynamics, n, torch_view(v))
            vf_fn = dynamics.get_vf_fn(**vf_kwargs)
            out = jax_view(vf_fn(t_torch, x_torch))
            for n, v in orig.items():
                object.__setattr__(dynamics, n, v)
            return out
        # base call with no extra arguments
        return jax_view(vf_fn(t_torch, x_torch))

    return _call_wrapped_drift_fn


def _get_diffusion_fn_wrapper(
    diffusion_fn: TDiffusion,
    df_kwargs: PyTree[Any] | None = None,
) -> Callable:
    # prepare argument in higer scope
    df_kwargs = df_kwargs or {}

    def _call_wrapped_diffusion_fn(t: TorchaxTensor, x: TorchaxTensor, args: TorchaxTensor | None = None) -> Callable:
        # handle backend
        t_torch = torch_view(t)
        x_torch = torch_view(x)
        x_jax = jax_view(x)

        # prepare arguments
        partial_diffusion, num_params = _prepare_diffusion_fn(diffusion_fn, df_kwargs)

        # call diffusion
        if num_params >= 2:
            out_torch = partial_diffusion(t_torch, x_torch)
        else:
            out_torch = partial_diffusion(t_torch)

        # handle output
        out_jax = jax_view(out_torch)
        if isinstance(out_jax, lx.AbstractLinearOperator):
            return out_jax
        if jnp.ndim(out_jax) == 0:
            return lx.DiagonalLinearOperator(jnp.full_like(x_jax, out_jax))
        if out_jax.shape == x_jax.shape:
            return lx.DiagonalLinearOperator(out_jax)
        if jnp.ndim(out_jax) == 2:
            return lx.MatrixLinearOperator(out_jax)
        return lx.PyTreeLinearOperator(out_jax, jax.eval_shape(lambda: x_jax))

    return _call_wrapped_diffusion_fn


def _init_wrapped_ode_dynamics(
    dynamics: TODEDynamics,
    vf_kwargs: dict[str, Any],
):
    wrapped_drift_fn = _get_drift_fn_wrapper(
        dynamics,
        vf_kwargs,
    )

    class _WrappedODEDynamics:
        def get_vf_fn(self, **kwargs):
            return wrapped_drift_fn

    return _WrappedODEDynamics()


def _init_wrapped_sde_terms(
    dynamics: TSDEDynamics,
    vf_kwargs: dict[str, Any],
    df_kwargs: dict[str, Any],
):
    # retrieve original terms
    drift_term, diffusion_term = dynamics

    # wrap functions
    drift_fn = _get_drift_fn_wrapper(
        drift_term,
        vf_kwargs=vf_kwargs,
    )
    diffusion_fn = _get_diffusion_fn_wrapper(diffusion_term, df_kwargs=df_kwargs)

    class _WrappedTerm:
        def __init__(self, fn):
            self.fn = fn

    # return wrapped terms
    return _WrappedTerm(drift_fn), _WrappedTerm(diffusion_fn)
