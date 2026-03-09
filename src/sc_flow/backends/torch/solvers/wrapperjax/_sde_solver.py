from typing import Any

import diffrax as dfx
import numpy as np
import torch
import torchax
from torch import Tensor
from torchax.interop import j2t_autograd, jax_view, torch_view

from sc_flow.backends.jax._utils import get_sde_solver
from sc_flow.backends.jax.solvers._sde_solver import SDESolver as JaxSDESolver
from sc_flow.backends.torch._types import TDevice, TSDEDynamics
from sc_flow.backends.torch.solvers._solver import BaseSolver
from sc_flow.backends.torch.solvers.wrapperjax._utils import _convert_to_torchax, _extract_differentiable_params
from sc_flow.backends.torch.solvers.wrapperjax._wrappers import _init_wrapped_sde_terms, _TorchaxToTensorDevice

__all__ = ["WrappedSDESolver"]


class WrappedSDESolver(BaseSolver[TSDEDynamics]):
    r"""Wrapped sde solver"""

    def __init__(
        self,
        dynamics: TSDEDynamics,
        *,
        method: str | None = None,
        device_id: TDevice = "cpu",
        vf_kwargs: dict[str, Any] | None = None,
        df_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__(dynamics=dynamics, method=method, device_id=device_id)
        torchax.enable_globally()

        drift_raw, diffusion_raw = dynamics
        _convert_to_torchax(drift_raw)

        drift_term, diffusion_term = _init_wrapped_sde_terms(
            dynamics,
            vf_kwargs=vf_kwargs,
            df_kwargs=df_kwargs,
        )
        self._drift_fn = drift_raw
        self._diffusion_fn = diffusion_raw

        drift_term = _convert_to_torchax(drift_term)
        diffusion_term = _convert_to_torchax(diffusion_term)

        # container for dynamics
        wrapped_sde_dynamics = (
            drift_term,
            diffusion_term.fn,
        )
        method = get_sde_solver(method)

        # instantiate solver class
        self._jax_solver = JaxSDESolver(
            dynamics=wrapped_sde_dynamics,
            method=method,
            device_id=device_id,
        )

    def solve(
        self,
        source: Tensor,
        t0: float = 0.0,
        t1: float = 1.0,
        brownian_motion: dfx.AbstractBrownianPath | None = None,
        bm_tol: float = 1e-3,
        *,
        num_time_steps: int = 500,
        return_trajectory: bool = False,
        solver_kwargs: dict[str, Any] | None = None,
    ) -> Tensor:
        """"""  # noqa
        source_tx = source.to("jax")

        drift_params_dict = _extract_differentiable_params(self._drift_fn)
        diffusion_params_dict = _extract_differentiable_params(self._diffusion_fn)

        drift_names = list(drift_params_dict.keys())
        diffusion_names = list(diffusion_params_dict.keys())

        drift_terms_tensors = list(drift_params_dict.values())
        diffusion_terms_tensors = list(diffusion_params_dict.values())

        requires_grad = torch.is_grad_enabled() and (
            source.requires_grad or any(p.requires_grad for p in drift_terms_tensors + diffusion_terms_tensors)
        )

        param_tensors = drift_terms_tensors + diffusion_terms_tensors
        param_names = drift_names + diffusion_names

        if requires_grad and param_tensors:
            source_tx = source.to("jax").requires_grad_(True)

            def _propagate_grad(grad):
                """Propagate gradients from the final state back to the source"""
                source.grad = torch.tensor(np.array(jax_view(grad)), dtype=source.dtype)

            source_tx.register_hook(_propagate_grad)

            def jax_solve_with_backprop(source_tx, *params_tx):
                source_jax = jax_view(source_tx)
                params_jax = [jax_view(p) for p in params_tx]
                solve_kwargs = {
                    **(solver_kwargs or {}),
                    "args": {
                        "params": params_jax,
                        "param_names": param_names,
                    },
                }
                trajectory = self._jax_solver.solve(
                    source_jax,
                    t0,
                    t1,
                    brownian_motion=brownian_motion,
                    bm_tol=bm_tol,
                    num_time_steps=num_time_steps,
                    return_trajectory=return_trajectory,
                    solver_kwargs=solve_kwargs,
                )
                return trajectory

            tsolver_fn = j2t_autograd(jax_solve_with_backprop)
            trajectory = tsolver_fn(source_tx, *param_tensors)
            return _TorchaxToTensorDevice.apply(trajectory, self._device)

        else:
            source_jax = source.to("jax")
            trajectory = self._jax_solver.solve(
                jax_view(source_jax),
                t0,
                t1,
                brownian_motion=brownian_motion,
                bm_tol=bm_tol,
                num_time_steps=num_time_steps,
                return_trajectory=return_trajectory,
                solver_kwargs=solver_kwargs,
            )
            trajectory = torch_view(trajectory).to(self._device)
            return trajectory
