import abc

import torch

from sckitflow._samplers import NoiseSamplerConfig
from sckitflow.core._utils import broadcast_to_target_shape

__all__ = ["BasePath", "OneSidedPath"]


class BasePath(abc.ABC):
    def __init__(
        self,
        sigma: float,
        sampler_cfg: NoiseSamplerConfig | None = None,
    ) -> None:
        # sanity check for positive values
        if not sigma >= 0:
            msg = f"Argument sigma should be a positive float for non deterministic probability paths. Found {sigma=}"
            raise ValueError(msg)
        if sigma > 0 and sampler_cfg is None:
            raise ValueError("Positive sigma requires a sampler to be passed")

        # setting the attributes
        self._sigma = sigma
        self._sampler = None if sampler_cfg is None else sampler_cfg.build()

    @abc.abstractmethod
    def compute_alpha_t(
        self,
        t: torch.Tensor,
    ) -> torch.Tensor: ...

    @abc.abstractmethod
    def compute_beta_t(
        self,
        t: torch.Tensor,
    ) -> torch.Tensor: ...

    @abc.abstractmethod
    def compute_sigma_t(
        self,
        t: torch.Tensor,
    ) -> torch.Tensor: ...

    def compute_alpha_t_and_dot_alpha_t(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        alpha_t, grad_fn = torch.func.vjp(self.compute_alpha_t, t)
        return alpha_t, grad_fn(torch.ones_like(alpha_t))[0]

    def compute_beta_t_and_dot_beta_t(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        beta_t, grad_fn = torch.func.vjp(self.compute_beta_t, t)
        return beta_t, grad_fn(torch.ones_like(beta_t))[0]

    def compute_sigma_t_and_dot_sigma_t(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sigma_t, grad_fn = torch.func.vjp(self.compute_sigma_t, t)
        return sigma_t, grad_fn(torch.ones_like(sigma_t))[0]

    def compute_dot_alpha_t(self, t: torch.Tensor) -> torch.Tensor:
        return self.compute_alpha_t_and_dot_alpha_t(t)[1]

    def compute_dot_beta_t(self, t: torch.Tensor) -> torch.Tensor:
        return self.compute_beta_t_and_dot_beta_t(t)[1]

    def compute_dot_sigma_t(self, t: torch.Tensor) -> torch.Tensor:
        return self.compute_sigma_t_and_dot_sigma_t(t)[1]

    def compute_xt(
        self,
        t: torch.Tensor,
        xt: torch.Tensor,
        x0: torch.Tensor,
        x1: torch.Tensor,
        *,
        generator: torch.Generator,
    ) -> torch.Tensor:
        alpha_t = broadcast_to_target_shape(self.compute_alpha_t(t), x0.shape)
        beta_t = broadcast_to_target_shape(self.compute_beta_t(t), x0.shape)
        mu_t = alpha_t * x0 + beta_t * x1
        if self._sigma > 0:
            sigma_t = broadcast_to_target_shape(self.compute_sigma_t(t), x0.shape)
            noise = self._sampler(x1.shape, generator=generator, device=x0.device, dtype=x0.dtype)
            return mu_t + sigma_t * noise
        return mu_t

    def compute_ut(
        self,
        t: torch.Tensor,
        xt: torch.Tensor,
        x0: torch.Tensor,
        x1: torch.Tensor,
        *,
        generator: torch.Generator,
    ) -> torch.Tensor:
        # Compute alpha terms.
        alpha_t, dot_alpha_t = self.compute_alpha_t_and_dot_alpha_t(t)
        alpha_t = broadcast_to_target_shape(alpha_t, x0.shape)
        dot_alpha_t = broadcast_to_target_shape(dot_alpha_t, x0.shape)

        # Compute beta terms.
        beta_t, dot_beta_t = self.compute_beta_t_and_dot_beta_t(t)
        beta_t = broadcast_to_target_shape(beta_t, x0.shape)
        dot_beta_t = broadcast_to_target_shape(dot_beta_t, x0.shape)

        # Compute mean and differential.
        mu_t = alpha_t * x0 + beta_t * x1
        dot_mu_t = dot_alpha_t * x0 + dot_beta_t * x1

        # Return mean differential for deterministic paths.
        if self._sigma <= 0:
            return dot_mu_t

        # Compute sigma terms.
        sigma_t, dot_sigma_t = self.compute_sigma_t_and_dot_sigma_t(t)
        sigma_t = broadcast_to_target_shape(sigma_t, x0.shape)
        dot_sigma_t = broadcast_to_target_shape(dot_sigma_t, x0.shape)

        # Guard for zero division
        eps = torch.finfo(sigma_t.dtype).eps
        safe_sigma_t = torch.where(sigma_t.abs() > eps, sigma_t, torch.ones_like(sigma_t))
        coeff = torch.where(
            sigma_t.abs() > eps,
            dot_sigma_t / safe_sigma_t,
            torch.zeros_like(sigma_t),
        )
        return dot_mu_t + coeff * (xt - mu_t)

    @property
    def sigma(self) -> float:
        return self._sigma


class OneSidedPath(BasePath):
    def compute_alpha_t(self, t: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(t)
