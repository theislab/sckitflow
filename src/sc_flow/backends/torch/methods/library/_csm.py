from typing import Any

import torch

from sc_flow.backends.torch._types import PredictionData
from sc_flow.backends.torch.coupling._coupling import independent_coupling
from sc_flow.backends.torch.methods._base import TorchGenerativeFlow
from sc_flow.backends.torch.methods._utils import StepData
from sc_flow.backends.torch.methods.library._utils import DenoiserVelocity
from sc_flow.backends.torch.nn._vf import BaseVelocityField, MLPVelocity
from sc_flow.backends.torch.probability_paths._probability_paths import SchrodingerBridgeProbabilityPath
from sc_flow.backends.torch.solvers import BaseSolver, ODESolver

__all__ = ["CSM"]


class CSM(TorchGenerativeFlow):
    _module_cls: type[BaseVelocityField] = MLPVelocity
    _default_solver_cls: type[BaseSolver] = ODESolver

    def __init__(self, *args, sigma_path: int = 1.0, **kwargs):
        super().__init__(*args, **kwargs)

        # set defaults
        if self._match_fn is None:
            self._match_fn = independent_coupling
        if self._noise_sampler is None:
            self._noise_sampler = torch.randn_like
        if self._time_sampler is None:
            self._time_sampler = torch.rand
        if self._probability_path is None:
            self._probability_path = SchrodingerBridgeProbabilityPath(sigma_path)

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
        # latent = torch.zeros_like(target)

        batch_size = latent.shape[0]
        t = self._time_sampler((batch_size,), device=latent.device, dtype=latent.dtype)
        xt = self._probability_path.compute_xt(t, latent, target)
        mu_t = self._probability_path.compute_mu_t(t, xt, latent)
        sigma_t = self._probability_path.compute_sigma_t(t)[..., None]
        sigma2_t = sigma_t**2
        cond = {
            **condition_data,
            **group_data,
        }
        st = self._module(t, xt, condition_dict=cond, source=source)
        loss = torch.nn.functional.mse_loss(st * sigma2_t, mu_t - xt)
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
        eps: float = 1e-15,
        tau: float = 1e-7,
        max_val: float = 10.0,
        **kwargs,
    ) -> PredictionData:
        # prepare latent state from step data
        if latent is None:
            latent = self._prepare_latent_state(step_data.source_state, step_data.target_state)
        # latent = torch.zeros_like(step_data.target_state)

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

        # initialize veocity
        vf = DenoiserVelocity(self._module, self._probability_path, eps=eps, max_val=max_val)

        # prepare solver and integrate dynamics
        if solver_cls is None:
            solver_cls = self._default_solver_cls
        time_grid = torch.linspace(tau, 1.0 - tau, steps=num_steps, device=latent.device, dtype=latent.dtype)
        solver = solver_cls(
            vf,
            *args,
            method=method,
            vf_kwargs={"latent": latent, "condition_dict": condition_dict, "source": step_data.source_state},
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
