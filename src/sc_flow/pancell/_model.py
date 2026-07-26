"""Pan-cell flow model: a foundation encoder in flow's **state-encoder slot** + a velocity field.

The composition that answers "different spaces → one space, then match flow": the shared
:class:`sc_flow.concept.GeneEncoder` maps raw-count cells from *any* gene panel into one latent, and the
velocity field transports source→target *in that latent*. Because the encoder is a
:class:`sc_flow.training.ArchitectureConfig` Component and the velocity is another, both are portable and
the encoder can be loaded pretrained + frozen or fine-tuned (its params are simply in / out of the optimizer).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from sc_flow.training._config import ArchitectureConfig

__all__ = ["VelocityMLP", "VelocityMLPConfig", "PanCellFlowModel"]

_ACT = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}


class VelocityMLP(nn.Module):
    """A velocity field ``v(z_t, t)`` over the latent space (torch-only; the OTFM/GENOT velocity is the
    same role with a jax/ott coupling in its objective)."""

    def __init__(self, dim: int, *, hidden: int = 256, n_layers: int = 3, time_dim: int = 32,
                 activation: str = "gelu") -> None:
        super().__init__()
        self.time_dim = time_dim - time_dim % 2  # even, for the sin/cos split
        act = _ACT[activation]
        layers: list[nn.Module] = [nn.Linear(dim + self.time_dim, hidden), act()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), act()]
        layers.append(nn.Linear(hidden, dim))
        self.net = nn.Sequential(*layers)

    def _time_embed(self, t: torch.Tensor) -> torch.Tensor:
        half = self.time_dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / max(half, 1))
        a = t * freqs  # (B, half)
        return torch.cat([torch.sin(a), torch.cos(a)], dim=-1)

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, self._time_embed(t)], dim=-1))


@dataclass
class VelocityMLPConfig(ArchitectureConfig, type_id="flow.velocity_mlp", version=1):
    """Portable recipe for :class:`VelocityMLP` (a :class:`sc_flow.Component`). ``dim`` = the latent width the
    state encoder produces."""

    dim: int
    hidden: int = 256
    n_layers: int = 3
    time_dim: int = 32
    activation: str = "gelu"

    def build(self, context: object = None) -> VelocityMLP:
        return VelocityMLP(self.dim, hidden=self.hidden, n_layers=self.n_layers, time_dim=self.time_dim,
                           activation=self.activation)


class PanCellFlowModel(nn.Module):
    """``state_encoder`` (raw-count cells → shared latent) + ``velocity`` (transport in the latent)."""

    def __init__(self, state_encoder: nn.Module, velocity: nn.Module) -> None:
        super().__init__()
        self.state_encoder = state_encoder
        self.velocity = velocity

    def encode(self, tokens: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.state_encoder(tokens, pad_mask)  # (B, dim) unnormalized latent
