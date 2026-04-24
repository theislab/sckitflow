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
from sc_flow.backends.torch.solvers._fm_solver import BaseSolver, FMSolver

__all__ = ["EMD"]


class EMD(TorchGenerativeFlow):
    _module_cls: type[BaseModule] = MLPFlowMap
    _default_solver_cls: type[BaseSolver] = FMSolver

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
            self._match_fn = cfm.method.match_fn
            self._noise_sampler = cfm.method.noise_sampler
            self._probability_path = cfm.method.probability_path
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

    def _step_fn(self, step_data: StepData, *args, **kwargs) -> tuple[torch.Tensor, dict[str, Any]]:
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
        s, t = self._time_sampler(
            (batch_size,),
            device=step_data.target_state.device,
            dtype=step_data.target_state.dtype,
        )

        # sample ground truth interpolant
        xs = self._probability_path.compute_xt(s, latent, step_data.target_state)

        # evaluate vf
        vf_fn = self.teacher_vf.get_vf_fn(cond, source=step_data.source_state)
        vs = vf_fn(s, xs)

        # forward pass on neural networks with jvp and compute time differential
        _, dXds = torch.func.jvp(
            self._module.get_vf_fn(cond, source=step_data.source_state),
            (s, t, xs),
            (torch.ones_like(s), torch.zeros_like(t), torch.zeros_like(xs)),
        )

        # forward pass on neural networks with jvp and compute time differential
        _, nablaXs_fn = torch.func.vjp(
            lambda xs: self._module.get_vf_fn(cond, source=step_data.source_state)(s, t, xs),
            xs,
        )
        nablaXs = nablaXs_fn(vs)[0]

        loss = torch.mean(self._weight_fn(s, t) * ((nablaXs + dXds) ** 2).sum(-1))
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

        # create solver instance with the condition dictionary and source
        solver = solver_cls(
            self._module,
            method=None,  # not used, kept for API
            device_id=self._device_id,
            vf_kwargs={"condition_dict": condition_dict, "source": step_data.source_state},
        )

        predictions = solver.solve(
            latent,
            time_grid,
            solver_kwargs=solver_kwargs,
            return_trajectory=return_trajectory,
        )

        if return_trajectory:
            samples = predictions[-1]
            traj = predictions
        else:
            samples = predictions
            traj = None

        return PredictionData(samples, traj=traj)

    @property
    def teacher_vf(self) -> BaseModule | None:
        if self._cfm is not None:
            return self._cfm.method.module
        else:
            return None
