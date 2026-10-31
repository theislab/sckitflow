import torch

from sckitflow._constants import PI
from sckitflow.core.probability_paths._base import BasePath, OneSidedPath

__all__ = [
    "LinearTwoSidedPath",
    "TrigTwoSidedPath",
    "EncDecTwoSidedPath",
    "LinearOneSidedPath",
    "TrigOneSidedPath",
    "VPOneSidedPath",
]


class _LinearPathMixin:
    def compute_beta_t(self, t: torch.Tensor) -> torch.Tensor:
        return t


class _TrigPathMixin:
    def compute_beta_t(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sin(0.5 * PI * t)


class LinearTwoSidedPath(_LinearPathMixin, BasePath):
    def compute_alpha_t(self, t: torch.Tensor) -> torch.Tensor:
        return 1 - t

    def compute_sigma_t(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.sigma * t * (1 - t))


class TrigTwoSidedPath(_TrigPathMixin, BasePath):
    def compute_alpha_t(self, t: torch.Tensor) -> torch.Tensor:
        return torch.cos(0.5 * PI * t)

    def compute_sigma_t(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.sigma * t * (1 - t))


class EncDecTwoSidedPath(BasePath):
    def compute_alpha_t(self, t: torch.Tensor) -> torch.Tensor:
        return torch.where(
            t <= 0.5,
            torch.cos(PI * t) ** 2,
            torch.zeros_like(t),
        )

    def compute_beta_t(self, t: torch.Tensor) -> torch.Tensor:
        return torch.where(
            t > 0.5,
            torch.cos(PI * t) ** 2,
            torch.zeros_like(t),
        )

    def compute_sigma_t(self, t: torch.Tensor) -> torch.Tensor:
        return self.sigma * torch.sin(PI * t) ** 2


class LinearOneSidedPath(_LinearPathMixin, OneSidedPath):
    def compute_sigma_t(self, t: torch.Tensor) -> torch.Tensor:
        return 1 - t


class TrigOneSidedPath(_TrigPathMixin, OneSidedPath):
    def compute_sigma_t(self, t: torch.Tensor) -> torch.Tensor:
        return torch.cos(0.5 * PI * t)


class VPOneSidedPath(_LinearPathMixin, OneSidedPath):
    def compute_sigma_t(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(1 - t**2)
