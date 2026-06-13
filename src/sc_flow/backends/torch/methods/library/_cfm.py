from typing import Any

import torch

from sc_flow.backends.torch._types import PredictionData
from sc_flow.backends.torch.coupling._coupling import independent_coupling
from sc_flow.backends.torch.methods._base import TorchGenerativeFlow
from sc_flow.backends.torch.methods._utils import StepData
from sc_flow.backends.torch.nn._vf import BaseVelocityField, MLPVelocity
from sc_flow.backends.torch.probability_paths._probability_paths import LinearDiracProbabilityPath
from sc_flow.backends.torch.solvers import BaseSolver, ODESolver

__all__ = ["CFM"]


class CFM(TorchGenerativeFlow):
    _module_cls: type[BaseVelocityField] = MLPVelocity
    _default_solver_cls: type[BaseSolver] = ODESolver

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # set defaults
        if self._match_fn is None:
            self._match_fn = independent_coupling
        if self._noise_sampler is None:
            self._noise_sampler = torch.randn_like
        if self._time_sampler is None:
            self._time_sampler = torch.rand
        if self._probability_path is None:
            self._probability_path = LinearDiracProbabilityPath()

    def _prepare_latent_state(
        self,
        source: torch.Tensor | None,
        target_reference: torch.Tensor,
    ) -> torch.Tensor:
        if source is None or self._generate_from_noise:
            return self._noise_sampler(target_reference)
        return source

    def _step_fn(self, step_data: StepData, *args, **kwargs) -> tuple[torch.Tensor, dict[str, Any]]:
        # extract step data
        target = step_data.target_state
        source = step_data.source_state
        condition_data = self._get_tensor_dict_from_data(step_data.target_condition_data)
        group_data = self._get_tensor_dict_from_data(step_data.target_group_data)

        latent = self._prepare_latent_state(source, target)
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

    # HACK: This is ugly but I can use multi-transition logic without breaking anything.
    # But I can write clean code once I figure out the differences between this and cont4extflow
    def train_step(self, matched_distr_or_nodes, *args, transitions=False, **kwargs):
        if transitions:
            return self._train_step_multi(matched_distr_or_nodes, *args, **kwargs)
        return super().train_step(matched_distr_or_nodes, *args, **kwargs)

    # multi-transition logic
    def _train_step_multi(self, nodes, *args, timepoints, **kwargs) -> tuple[torch.Tensor, dict[str, Any]]:
        all_t, all_xt, all_ut, all_cond, all_latent = [], [], [], [], []

        for i, node in enumerate(nodes):
            step_data = self._extract_step_data(node)
            step_data = self._extract_matched_observations(step_data)  # OT per transition

            target = step_data.target_state
            source = step_data.source_state
            condition_data = self._get_tensor_dict_from_data(step_data.target_condition_data)
            group_data = self._get_tensor_dict_from_data(step_data.target_group_data)

            latent = self._prepare_latent_state(source, target)
            batch_size = latent.shape[0]

            t_start = timepoints[i]
            t_end = timepoints[i + 1]
            delta_t = t_end - t_start
            t_local = self._time_sampler((batch_size,), device=latent.device, dtype=latent.dtype) * delta_t
            t_global = t_local + t_start

            xt = self._probability_path.compute_xt(t_local, latent, target)
            ut = self._probability_path.compute_ut(t_local, xt, latent, target)

            cond = {
                **condition_data,
                **group_data,
            }

            all_t.append(t_global)
            all_xt.append(xt)
            all_ut.append(ut)
            all_cond.append(cond)
            all_latent.append(latent)

        # single forward pass over all transitions
        t_all = torch.cat(all_t)
        xt_all = torch.cat(all_xt)
        ut_all = torch.cat(all_ut)
        latent_all = torch.cat(all_latent)
        cond_all = {k: torch.cat([c[k] for c in all_cond]) for k in all_cond[0]}

        vt_all = self._module(t_all, xt_all, condition_dict=cond_all, source=latent_all)
        loss = torch.nn.functional.mse_loss(vt_all, ut_all)

        return loss, {"loss": loss.item()}

    def _predict(
        self,
        step_data: StepData,
        *args,
        solver_cls: type[BaseSolver] | None = None,
        solver_kwargs: dict[str, Any] | None = None,
        return_trajectory: bool = False,
        num_steps: int = 100,
        latent: torch.Tensor | None = None,
        t_start: float = 0.0,
        t_end: float = 1.0,
        **kwargs,
    ) -> PredictionData:
        # prepare latent state from step data
        if latent is None:
            latent = self._prepare_latent_state(step_data.source_state, step_data.target_state)

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
            solver_cls = self._default_solver_cls
        time_grid = torch.linspace(t_start, t_end, steps=num_steps, device=latent.device, dtype=latent.dtype)
        solver = solver_cls(
            self._module,
            *args,
            method=method,
            vf_kwargs={"condition_dict": condition_dict, "source": step_data.source_state},
            device_id=self._device_id,
            **kwargs,
        )
        predictions = solver.solve(
            latent,
            time_grid,
            solver_kwargs=solver_kwargs,
            return_trajectory=return_trajectory,
        )

        # split samples and trajectories
        if return_trajectory:
            samples = predictions[-1]
            traj = predictions
        else:
            samples = predictions
            traj = None

        # define prediction data
        return PredictionData(
            samples,
            traj=traj,
        )
