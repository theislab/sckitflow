import torch

from sc_flow.backends.torch.methods._base_methods import BaseGenerativeFlow
from sc_flow.backends.torch.nn._vf import BaseVelocityField, MLPVelocity

__all__ = ["FlowMatching"]


class FlowMatching(BaseGenerativeFlow):
    _module_cls: type[BaseVelocityField] = MLPVelocity

    def _compute_loss(self, latent, source, target, condition_data, group_data):
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

        return loss

    def predict(self, cond, latent, source, return_trajectory: bool = False):
        solver_kwargs = dict(self._solver_kwargs)
        method = solver_kwargs.pop("method", None)
        time_grid = torch.linspace(0.0, 1.0, steps=2, device=latent.device, dtype=latent.dtype)
        solver = self.solver_cls(
            self._module,
            method=method,
            vf_kwargs={"condition_dict": cond, "source": source},
        )
        return solver.solve(latent, time_grid, solver_kwargs=solver_kwargs, return_trajectory=return_trajectory)
