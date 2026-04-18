from typing import Any

import torch

from sc_flow.backends.torch.methods._base import BaseGenerativeFlow
from sc_flow.backends.torch.methods._utils import StepData
from sc_flow.backends.torch.nn._vf import BaseVelocityField, MLPVelocity
from sc_flow.backends.torch.solvers import BaseSolver, ODESolver

__all__ = ["CFM"]


class CFM(BaseGenerativeFlow):
    _module_cls: type[BaseVelocityField] = MLPVelocity

    def _compute_loss(
        self,
        latent: torch.Tensor,
        source: torch.Tensor | None,
        target: torch.Tensor | None,
        condition_data: dict[str, torch.Tensor] | None,
        group_data: dict[str, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        batch_size = latent.shape[0]
        t = self._time_sampler((batch_size,), device=latent.device, dtype=latent.dtype)
        xt = self._probability_path.compute_xt(t, latent, target)
        ut = self._probability_path.compute_ut(t, xt, latent, target)
        cond = {
            **condition_data,
            **group_data,
        }
        vt = self._module(t, xt, condition_dict=cond, source=source)
        loss = torch.nn.functional.mse_loss(vt, ut)

        return loss, {"loss": loss.item()}

    def _predict(
        self,
        latent: torch.Tensor,
        step_data: StepData,
        solver_cls: type[BaseSolver] | None = None,
        solver_kwargs: dict[str, Any] | None = None,
        return_trajectory: bool = False,
        num_steps: bool = 100,
    ):
        # extract condition and groups data
        condition_reps_dict = self._get_tensor_dict_from_data(step_data.target_condition_data)
        group_reps_dict = self._get_tensor_dict_from_data(step_data.target_group_data)

        # initialize condition dict
        condition_dict = {
            **condition_reps_dict,
            **group_reps_dict,
        }

        # handle solver kwargs
        if solver_kwargs is None:
            solver_kwargs = {}
        solver_kwargs.setdefault("method", "euler")
        method = solver_kwargs.pop("method")

        # prepare solver and integrate dynamics
        if solver_cls is None:
            solver_cls = ODESolver
        time_grid = torch.linspace(0.0, 1.0, steps=num_steps, device=latent.device, dtype=latent.dtype)
        solver = solver_cls(
            self._module,
            method=method,
            vf_kwargs={"condition_dict": condition_dict, "source": step_data.source_state},
            device_id=self._device_id,
        )
        return solver.solve(
            latent,
            time_grid,
            solver_kwargs=solver_kwargs,
            return_trajectory=return_trajectory,
        )
