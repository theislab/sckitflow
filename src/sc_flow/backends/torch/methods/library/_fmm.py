from collections.abc import Callable
from typing import Any

import torch

from sc_flow.backends.torch._types import PredictionData
from sc_flow.backends.torch.coupling._coupling import independent_coupling
from sc_flow.backends.torch.methods._base import TorchGenerativeFlow
from sc_flow.backends.torch.methods._utils import StepData
from sc_flow.backends.torch.methods.library._cfm import CFM
from sc_flow.backends.torch.nn._fm import MLPFlowMap
from sc_flow.backends.torch.nn._modules import BaseModule
from sc_flow.backends.torch.probability_paths._probability_paths import LinearDiracProbabilityPath
from sc_flow.backends.torch.solvers import BaseSolver

__all__ = ["FMM"]


class FMM(TorchGenerativeFlow):
    _module_cls: type[BaseModule] = MLPFlowMap
    _default_solver_cls: type[BaseSolver] = None

    def __init__(
        self,
        *args,
        cfm: CFM | None = None,
        weight_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        # distillation from teacher CFM model, in this case
        # take necessary attributes from cfm for compatibility
        if cfm is not None:
            self._match_fn = cfm.match_fn
            self._noise_sampler = cfm.noise_sampler
            self._probability_path = cfm.probability_path
        else:
            # set defaults
            if self._match_fn is None:
                self._match_fn = independent_coupling
            if self._noise_sampler is None:
                self._noise_sampler = torch.randn_like
            if self._probability_path is None:
                self._probability_path = LinearDiracProbabilityPath()

        # set default time sampler
        if self._time_sampler is None:

            def _time_sampler(*args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
                s = torch.rand(*args, **kwargs)
                t = torch.rand(*args, **kwargs)
                return s, t

            self._time_sampler = _time_sampler

        # set default weight function
        if weight_fn is None:

            def _weight_fn(
                s: torch.Tensor,
                t: torch.Tensor,
            ) -> torch.Tensor:
                return torch.ones_like(s)
        else:
            _weight_fn = weight_fn
        self._weight_fn = _weight_fn

        # register cfm
        self._cfm = cfm

    def _prepare_latent_state(
        self,
        source: torch.Tensor | None,
        target_reference: torch.Tensor,
    ) -> torch.Tensor:
        if source is None or self._generate_from_noise:
            return self._noise_sampler(target_reference)
        return source

    def _compute_loss_distillation(self, step_data: StepData, *args, **kwargs) -> tuple[torch.Tensor, dict[str, Any]]:
        # prepare condition
        condition_data = self._get_tensor_dict_from_data(step_data.target_condition_data)
        group_data = self._get_tensor_dict_from_data(step_data.target_group_data)
        cond = {
            **condition_data,
            **group_data,
        }

        # prepare latent state from step data
        latent = self._prepare_latent_state(step_data.source_state, step_data.target_state)

        # retrieving batch size and ode time
        batch_size = step_data.target_state.shape[0]
        s, t = self.time_sampler((batch_size,), device=step_data.target_state.device)

        # sample ground truth interpolant
        xs = self._probability_path.compute_xt(s, latent, step_data.target_state)

        # forward pass on neural networks with jvp
        xts_hat, dXdt = torch.func.jvp(
            self._module.get_vf_fn(cond, source=step_data.source_state),
            (s, t, xs),
            (torch.zeros_like(s), torch.ones_like(t), torch.zeros_like(xs)),
        )
        # evaluate vf
        vf_fn = self._cfm._module.get_vf_fn(cond, source=step_data.source_state)
        vt = vf_fn(t, xts_hat)
        loss = torch.mean(self._weight_fn(s, t) * ((dXdt - vt) ** 2).sum(-1))
        return loss, {"loss": loss.item()}

    def _compute_loss_e2e(self, step_data: StepData, *args, **kwargs) -> tuple[torch.Tensor, dict[str, Any]]:
        # prepare condition
        condition_data = self._get_tensor_dict_from_data(step_data.target_condition_data)
        group_data = self._get_tensor_dict_from_data(step_data.target_group_data)
        cond = {
            **condition_data,
            **group_data,
        }

        # prepare latent state from step data
        latent = self._prepare_latent_state(step_data.source_state, step_data.target_state)

        # retrieving batch size and ode time
        batch_size = step_data.target_state.shape[0]
        s, t = self.time_sampler((batch_size,), device=step_data.target_state.device)

        # sample ground truth interpolant and compute corresponding velocity field
        xt = self._probability_path.compute_xt(t, latent, step_data.target_state)
        ut = self._probability_path.compute_ut(t, latent, step_data.target_state, xt)

        # forward pass on neural networks
        xst_hat = self._module(t, s, xt, cond, source=step_data.source_state)
        _, dXdt = torch.func.jvp(
            self._module.get_vf_fn(cond, source=step_data.source_state),
            (s, t, xst_hat),
            (torch.zeros_like(s), torch.ones_like(t), torch.zeros_like(xst_hat)),
        )

        loss = torch.mean(self._weight_fn(s, t) * ((dXdt - ut) ** 2).sum(-1))
        return loss, {"loss": loss.item()}

    def _step_fn(self, step_data: StepData, *args, **kwargs) -> tuple[torch.Tensor, dict[str, Any]]:
        # distill from flow model when provided
        if self._cfm is not None:
            return self._compute_loss_distillation(step_data, *args, **kwargs)
        return self._compute_loss_e2e(step_data, *args, **kwargs)

    def _predict(
        self,
        step_data: StepData,
        *args,
        solver_cls: type[BaseSolver] | None = None,
        solver_kwargs: dict[str, Any] | None = None,
        return_trajectory: bool = False,
        num_steps: int = 100,
        latent: torch.Tensor | None = None,
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

        # prepare solver and integrate dynamics
        if solver_cls is None:
            solver_cls = self._default_solver_cls
        time_grid = torch.linspace(0.0, 1.0, steps=num_steps + 1, device=latent.device, dtype=latent.dtype)

        # get map fn
        map_fn = self.module.get_vf_fn(  # TODO: rename this
            condition_dict, source=step_data.source_state
        )

        # simulate flow map
        X_s = latent
        traj = [X_s]
        for idx, s in enumerate(time_grid[:-1]):
            t = time_grid[idx + 1]
            s_tensor = torch.ones([*latent.shape[:-1]], device=latent.device).float() * s
            t_tensor = torch.ones([*latent.shape[:-1]], device=latent.device).float() * t
            X_s = map_fn(s_tensor, t_tensor, X_s)
            traj.append(X_s)
        predictions = torch.stack(traj, axis=0)

        # split samples and trajectories
        if return_trajectory:
            samples = predictions[-1]
            traj = predictions
        else:
            samples = predictions[-1]
            traj = None

        # define prediction data
        return PredictionData(
            samples,
            traj=traj,
        )
