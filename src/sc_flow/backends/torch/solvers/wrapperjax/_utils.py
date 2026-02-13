from collections import OrderedDict
from typing import Any

import diffrax as dfx
import torchax
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
    "implicit_adams": dfx.Kvaerno5(),
    "fixed_adams": dfx.Tsit5(),
}


def _pack_state(module: nn.Module) -> Tensor:
    state = OrderedDict()
    for name, param in module.named_parameters(recurse=True):
        state[name] = param

    for name, buffer in module.named_buffers(recurse=True):
        state[name] = buffer

    return state


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


def dynamics_wrapper(dynamics: TODEDynamics) -> JAXTODEdynamics:
    torchax.enable_globally()

    class DynamicsWrapper:
        def __init__(self, dyn):
            self._dyn = dyn

        def get_vf_fn(self, **vf_kwargs: Any):
            vf_kwargs = vf_kwargs or {}
            vf_kwargs = _to_jax_tree(vf_kwargs)
            vf_fn = self._dyn.get_vf_fn(**vf_kwargs)

            def jax_vf(t, x, args=None):
                t_torch = _to_jax_tree(torch_view(t))
                x_torch = _to_jax_tree(torch_view(x))

                out_torch = vf_fn(t_torch, x_torch)

                out_jax = jax_view(_to_jax_tree(out_torch))

                return out_jax

            return jax_vf

    return DynamicsWrapper(dynamics)
