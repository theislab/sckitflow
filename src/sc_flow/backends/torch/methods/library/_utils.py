import torch

from sc_flow.backends.torch._types import MappedTensor, TVfFn
from sc_flow.backends.torch.nn import BaseVelocityField
from sc_flow.backends.torch.probability_paths import LinearProbabilityPath

__all__ = ["DenoiserVelocity"]


class DenoiserVelocity:
    def __init__(
        self,
        score_net: BaseVelocityField,
        probability_path: LinearProbabilityPath,
        eps: float = 1e-15,
        max_val: float = 10.0,
    ) -> None:
        self._score_net = score_net
        self._probability_path = probability_path
        self._eps = eps
        self._max_val = max_val

    def forward(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        latent: torch.Tensor,
        condition_dict: MappedTensor | None = None,
        source: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # compute probability path coefficients
        alpha = self._probability_path.compute_alpha_t(t)
        sigma = self._probability_path.compute_sigma_t(t)
        alpha_dot = self._probability_path.compute_dot_alpha_t(t)
        sigma_dot = self._probability_path.compute_dot_sigma_t(t)

        # compute coefficients to rescale score and evaluate score network
        num = sigma * (alpha * sigma_dot - alpha_dot * sigma)
        den = alpha
        coef = num / (den + self._eps)
        coef = torch.clamp(coef, max=self._max_val)
        print(f"{t=} -> {coef=}")
        st = self._score_net(t, x, condition_dict=condition_dict, source=source)

        # compute base drift term
        coef_base = alpha_dot / (alpha + self._eps)
        coef_base = torch.clamp(coef_base, min=-1.0 * self._max_val, max=self._max_val)
        print(f"{t=} -> {coef_base=}")
        vt_base = alpha_dot / (alpha + self._eps) * (x - latent)
        return vt_base + coef * st

    def get_vf_fn(
        self,
        latent: torch.Tensor,
        condition_dict: MappedTensor | None = None,
        source: torch.Tensor | None = None,
    ) -> TVfFn:
        """Compiles the velocity field function to be fed to external solvers."""

        def _vf_fn(t: torch.Tensor, x: torch.Tensor):
            return self.forward(t, x, latent, condition_dict=condition_dict, source=source)

        return _vf_fn
