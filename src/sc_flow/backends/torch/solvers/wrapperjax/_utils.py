from collections import OrderedDict
from typing import Any

import diffrax as dfx
from torch import Tensor, nn
from torchax.interop import jax_view, torch_view

from sc_flow.backends.jax._types import TODEDynamics as JAXTODEdynamics
from sc_flow.backends.torch._types import TODEDynamics

_ODE_SOLVER_REGISTRY = {
    "euler": dfx.Euler(),
    "midpoint": dfx.Midpoint(),
    "bosh3": dfx.Bosh3(),
    "dopri5": dfx.Dopri5(),
    "dopri8": dfx.Dopri8(),
    "rk4": dfx.Dopri5(),
    "heun2": dfx.Heun(),
    "heun3": dfx.Heun(),
    "adaptive_heun": dfx.Heun(),
    "fehlberg2": dfx.Bosh3(),
    "explicit_adams": dfx.Tsit5(),
    "fixed_adams": dfx.Tsit5(),
}


def _to_jax_tree(x: Any) -> Any:
    if isinstance(x, Tensor):
        return x if x.device.type == "jax" else x.to("jax")
    elif isinstance(x, dict):
        return {k: _to_jax_tree(v) for k, v in x.items()}
    elif isinstance(x, (list | tuple)):
        return type(x)(_to_jax_tree(v) for v in x)
    return x


def map_torch_method_to_jax(method: str | None) -> dfx.AbstractSolver:
    if method is None:
        return dfx.Euler()
    key = method.lower()
    if key not in _ODE_SOLVER_REGISTRY:
        raise KeyError(f"Unknown ODE solver method: {method!r}. Available: {sorted(_ODE_SOLVER_REGISTRY)}")
    return _ODE_SOLVER_REGISTRY[key]


def _extract_differentiable_params(obj: Any) -> OrderedDict[str, Tensor]:
    """Return an ordered dict of (name -> live Parameter/Tensor) from *obj*.

    Deterministic ordering is essential: we flatten these into positional
    args for jax.vjp and must be able to reconstruct them later.
    """
    params: OrderedDict[str, Tensor] = OrderedDict()

    if isinstance(obj, nn.Module):
        for name, p in obj.named_parameters(recurse=True):
            params[name] = p
        for name, b in obj.named_buffers(recurse=True):
            if b.requires_grad:
                params[name] = b
    else:
        for attr_name in sorted(dir(obj)):
            if attr_name.startswith("_"):
                continue
            try:
                attr = getattr(obj, attr_name)
                if isinstance(attr, nn.Parameter):
                    params[attr_name] = attr
                elif isinstance(attr, Tensor) and attr.requires_grad:
                    params[attr_name] = attr
            except (AttributeError, TypeError):
                pass

    return params


def dynamics_wrapper(dynamics: TODEDynamics) -> JAXTODEdynamics:
    class DynamicsWrapper:
        def __init__(self, dyn):
            self._dyn = dyn

        def get_vf_fn(self, **vf_kwargs: Any):
            vf_kwargs = vf_kwargs or {}

            def jax_vf(t, x, args=None):
                t_torch = torch_view(t)
                x_torch = torch_view(x)

                if args is not None and "params" in args:
                    params_jax = args["params"]
                    param_names = args["param_names"]

                    orig = {n: getattr(self._dyn, n) for n in param_names}
                    for n, v in zip(param_names, params_jax, strict=False):
                        object.__setattr__(self._dyn, n, torch_view(v))
                    vf_fn = self._dyn.get_vf_fn(**vf_kwargs)
                    out = jax_view(vf_fn(t_torch, x_torch))
                    for n, v in orig.items():
                        object.__setattr__(self._dyn, n, v)
                    return out
                else:
                    vf_fn = self._dyn.get_vf_fn(**vf_kwargs)
                    return jax_view(vf_fn(t_torch, x_torch))

            return jax_vf

    return DynamicsWrapper(dynamics)
