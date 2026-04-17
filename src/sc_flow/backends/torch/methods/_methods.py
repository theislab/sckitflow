import warnings
from typing import Any

import torch

from sc_flow.backends.jax.probability_paths._probability_paths import BaseProbabilityPath
from sc_flow.backends.torch._types import MappedTensor, TMatchFn, TNoiseSamplerFn, TTimeSamplerFn
from sc_flow.backends.torch.methods._base_methods import BaseGenerativeFlow
from sc_flow.backends.torch.nn._modules import BaseModule
from sc_flow.backends.torch.nn._vf import BaseVelocityField
from sc_flow.backends.torch.solvers.solver import BaseSolver, ODESolver

__all__ = ["GENOT"]


class FlowMatching(BaseGenerativeFlow):
    def __init__(
        self,
        module: BaseVelocityField,
        probability_path: BaseProbabilityPath,
        match_fn: TMatchFn,
        noise_sampler: TNoiseSamplerFn,
        time_sampler: TTimeSamplerFn,
        generate_from_noise: bool = False,
        solver_cls: type[BaseSolver] = ODESolver,
        solver_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            module,
            probability_path,
            match_fn,
            solver_cls,
            noise_sampler,
            time_sampler,
            generate_from_noise,
        )
        self._solver_kwargs = {} if solver_kwargs is None else solver_kwargs

    def _compute_loss(self, latent, source, target, condition_data, group_data):
        if self.generate_from_noise:
            if source is not None:
                warnings.warn("Source data should be None when generate_from_noise is True.", stacklevel=2)
                latent = torch.randn_like(target, dtype=target.dtype)
            else:
                latent = source

        batch_size = latent.shape[0]
        t = self._time_sampler(batch_size, device=latent.device, dtype=latent.dtype)
        xt = self._probability_path.compute_xt(t, latent, target)
        ut = self._probability_path.compute_ut(t, xt, latent, target)
        # TODO: fix this to be the whole data without splits
        cond = {
            **condition_data,
            **group_data,
        }
        vt = self._module(xt, t, cond, latent)
        loss = torch.nn.functional.mse_loss(vt, ut)

        return loss

    def _predict_cond(self, cond, latent, source, return_trajectory: bool = False):
        solver_kwargs = dict(self._solver_kwargs)
        method = solver_kwargs.pop("method", None)
        time_grid = torch.linspace(0.0, 1.0, steps=2, device=latent.device, dtype=latent.dtype)
        solver = self.solver_cls(
            self._module,
            method=method,
            vf_kwargs={"condition_dict": cond, "source": source},
        )
        return solver.solve(latent, time_grid, solver_kwargs=solver_kwargs, return_trajectory=return_trajectory)

    def predict(self, source, condition_data, group_data, return_trajectory: bool = False):
        prediction = []

        if source is None and not self.generate_from_noise:
            raise ValueError("Source data cannot be None when generate_from_noise is False.", stacklevel=2)

        if self.generate_from_noise:
            latent = self.noise_sampler(source)
        else:
            latent = source

        for cond in condition_data:
            prediction.append(self._predict_cond(cond, latent, source, return_trajectory))

        return prediction


# do stochastic flow matching as a child class where it takes the diffusion function for score matching


class GENOT(BaseGenerativeFlow):
    def __init__(
        self,
        module: BaseVelocityField,
        probability_path: BaseProbabilityPath,
        match_fn: TMatchFn,
        noise_sampler: TNoiseSamplerFn,
        time_sampler: TTimeSamplerFn,
        target_dim: int,
        generate_from_noise: bool = False,
        solver_cls: type[BaseSolver] = ODESolver,
        solver_kwargs: dict[str, Any] | None = None,
        latent_noise_fn: Any | None = None,
    ) -> None:
        super().__init__(
            module,
            probability_path,
            match_fn,
            solver_cls,
            noise_sampler,
            time_sampler,
            generate_from_noise,
        )
        self._solver_kwargs = {} if solver_kwargs is None else solver_kwargs
        if latent_noise_fn is None:
            self._latent_noise_fn = lambda x: torch.randn(x.shape[0], target_dim, device=x.device, dtype=x.dtype)
        else:
            self._latent_noise_fn = latent_noise_fn

    def _compute_loss(self, latent, source, target, condition_data, group_data):
        if self.generate_from_noise:
            if source is not None:
                warnings.warn("Source data should be None when generate_from_noise is True.", stacklevel=2)
                latent = torch.randn_like(target, dtype=target.dtype)
            else:
                latent = source

        batch_size = latent.shape[0]
        t = self._time_sampler(batch_size, device=latent.device, dtype=latent.dtype)
        xt = self._probability_path.compute_xt(t, latent, target)
        ut = self._probability_path.compute_ut(t, xt, latent, target)
        # TODO: change this after refactoring
        cond = {
            **condition_data,
            **group_data,
        }
        v_t = self._module(xt, t, cond, latent)
        loss = torch.mean((v_t - ut) ** 2)

        return loss

    def _compute_loss_a(
        self,
        eta_network: BaseModule,
        x: torch.Tensor,
        a: torch.Tensor,
        expectation_reweighting: float,
    ):
        eta_predictions = eta_network.forward(x)
        loss = torch.mean((eta_predictions - a) ** 2) + (torch.mean(eta_predictions) - expectation_reweighting) ** 2
        return loss

    def _compute_loss_b(
        self,
        eta_network: BaseModule,
        x: torch.Tensor,
        b: torch.Tensor,
        expectation_reweighting: float,
    ):
        eta_predictions = eta_network.forward(x)
        loss = torch.mean((eta_predictions - b) ** 2) + (torch.mean(eta_predictions) - expectation_reweighting) ** 2
        return loss

    def _predict_single(
        self,
        source: torch.Tensor,
        condition: MappedTensor | None,
        *,
        return_trajectory: bool = False,
    ) -> torch.Tensor:
        """Solve the learned ODE for one condition group.

        Samples a latent in the target space and integrates from t=0 to t=1,
        with ``source`` and ``condition`` held fixed as VF context.
        """
        latent = self._latent_noise_fn(source)
        time_grid = torch.linspace(0.0, 1.0, steps=2, device=source.device, dtype=source.dtype)

        solver_kwargs = dict(self._solver_kwargs)
        method = solver_kwargs.pop("method", None)

        solver = self.solver_cls(
            self._module,
            method=method,
            vf_kwargs={"condition_dict": condition, "source": source},
        )
        return solver.solve(latent, time_grid, solver_kwargs=solver_kwargs, return_trajectory=return_trajectory)

    def predict(
        self,
        source: dict[str, torch.Tensor],
        condition_data: dict[str, MappedTensor] | None = None,
        *,
        return_trajectory: bool = False,
    ) -> dict[str, torch.Tensor]:
        self._module.eval()
        predictions: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for key, src in source.items():
                cond = condition_data.get(key) if condition_data is not None else None
                predictions[key] = self._predict_single(src, cond, return_trajectory=return_trajectory)
        return predictions
