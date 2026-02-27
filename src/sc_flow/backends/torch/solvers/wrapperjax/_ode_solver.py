from typing import Any

import torch
import torchax
from torch import Tensor, nn
from torchax.interop import j2t_autograd, jax_view, torch_view

from sc_flow.backends.jax.solvers import ODESolver as JAXODESolver
from sc_flow.backends.torch._types import TDevice, TODEDynamics
from sc_flow.backends.torch.solvers.solver import BaseSolver
from sc_flow.backends.torch.solvers.wrapperjax._utils import _extract_differentiable_params, _map_torch_method_to_jax
from sc_flow.backends.torch.solvers.wrapperjax._wrappers import _init_wrapped_ode_dynamics

__all__ = ["WrappedODESolver"]


class WrappedODESolver(BaseSolver[TODEDynamics]):
    r"""Class for solving deterministic ordinary differential equations (ODEs) with :func:`torchdiffeq.odeint`.

    :param dynamics: Velocity field providing the time-dependent dynamics. Must implement :meth:`BaseVelocityField.get_vf_fn`.
    :type dynamics: class:`TODEDynamics`

    :param method: (Optional) Integration scheme used by ``torchdiffeq``. Defaults to ``"euler"``. Other valid options depend on ``torchdiffeq``.
    :type method: class:`str`

    :param device_id: (Optional) Identifier for the target compute device.
    :type device_id: class:`TDevice`

    :param vf_kwargs: (Optional) Keyword arguments passed to
            :meth:`BaseVelocityField.get_vf_fn`.
        :type vf_kwargs: class:`dict[str, Any] | None`
    """

    def __init__(
        self,
        dynamics: TODEDynamics,
        *,
        method: str = "euler",
        device_id: TDevice = "cpu",
        vf_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(dynamics=dynamics, method=method, device_id=device_id)
        torchax.enable_globally()
        vf_kwargs = vf_kwargs or {}
        if isinstance(dynamics, nn.Module):
            dynamics.to("jax")
        else:
            for attr_name in vars(dynamics):
                attr = getattr(dynamics, attr_name)
                if isinstance(attr, nn.Parameter):
                    setattr(dynamics, attr_name, nn.Parameter(attr.to("jax"), requires_grad=attr.requires_grad))

        _jax_dynamics = _init_wrapped_ode_dynamics(dynamics, vf_kwargs)

        self._dyn = dynamics

        jax_method = _map_torch_method_to_jax(method)

        self._jax_solver = JAXODESolver(
            dynamics=_jax_dynamics,
            method=jax_method,
            device_id=device_id,
            vf_kwargs=vf_kwargs,
        )

    def solve(
        self,
        source: Tensor,
        t0: float = 0.0,
        t1: float = 1.0,
        *,
        num_time_steps: int = 500,
        return_trajectory: bool = False,
        solver_kwargs: dict[str, Any] | None = None,
    ) -> Tensor:
        r"""Integrates the ODE defined by the provided velocity field.

        :param source: Initial state of the ODE.
        :type source: class:`torch.Tensor`

        :param t0: Initial time of the ODE.
        :type t0: class:`torch.Tensor`

        :param t1: Final time of the ODE.
        :type t1: class:`torch.Tensor`

        :param num_time_steps: Number of time steps to use for integration.
        :type num_time_steps: class:`int`

        :param solver_kwargs: (Optional) Keyword arguments forwarded directly to
            :func:`torchdiffeq.odeint`.
        :type solver_kwargs: class:`dict[str, Any] | None`

        :param return_trajectory: When ``True``, returns the full trajectory, else returns only the final state at the last time point.
        :type return_trajectory: class:`bool`

        :returns: Either the full state trajectory or the final state depending on
            :param:`return_trajectory`.
        :rtype: class:`torch.Tensor`
        """
        solver_kwargs = solver_kwargs or {}
        stepsize_controller = solver_kwargs.get("stepsize_controller", None)
        stepsize_controller_dfx = stepsize_controller.get_controller() if stepsize_controller else None

        solver_kwargs["stepsize_controller"] = stepsize_controller_dfx

        params_dict = _extract_differentiable_params(self._dyn)
        param_names = list(params_dict.keys())
        param_tensors = list(params_dict.values())

        requires_grad = torch.is_grad_enabled() and (
            source.requires_grad or any(p.requires_grad for p in param_tensors)
        )

        if requires_grad and param_tensors:
            source_torchax = source.to("jax")

            def jax_solve_with_backprop(source_torchax, *params_torchax):
                source_jax = jax_view(source_torchax)
                params_jax = [jax_view(p) for p in params_torchax]
                solve_kwargs = {
                    **solver_kwargs,
                    "args": {
                        "params": params_jax,
                        "param_names": param_names,
                    },
                }
                trajectory = self._jax_solver.solve(
                    source_jax,
                    t0,
                    t1,
                    num_time_steps=num_time_steps,
                    return_trajectory=True,
                    solver_kwargs=solve_kwargs,
                )
                return trajectory, trajectory[-1]

            tsolver_fn = j2t_autograd(jax_solve_with_backprop)
            trajectory, final_state = tsolver_fn(source_torchax, *param_tensors)
            print("result:", final_state.requires_grad, getattr(final_state, "grad_fn", None))

            if return_trajectory:
                return trajectory
            else:
                print("result[-1]:", final_state.requires_grad, getattr(final_state, "grad_fn", None))
                return final_state
        else:
            source_jax = source.to("jax")
            trajectory = self._jax_solver.solve(
                jax_view(source_jax),
                t0,
                t1,
                num_time_steps=num_time_steps,
                return_trajectory=return_trajectory,
                solver_kwargs=solver_kwargs,
            )
            trajectory = torch_view(trajectory)
            if return_trajectory:
                return trajectory.to(self._device)
            else:
                return trajectory[-1].to(self._device)

    @property
    def jax_solver(self) -> JAXODESolver:
        """Access the underlying JAX ODE solver instance."""
        return self._jax_solver
