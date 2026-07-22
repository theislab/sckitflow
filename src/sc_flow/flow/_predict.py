
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from scfit.training._predictor import Predictor, register_predictor

__all__ = ["integrate_translation", "condition_mask_to_device", "condition_to_device", "ODEPredictor"]


def _as_f32(v: Any, device: Any) -> torch.Tensor:
    if isinstance(v, torch.Tensor):
        return v.to(device=device, dtype=torch.float32)
    return torch.as_tensor(np.asarray(v, dtype=np.float32), device=device)


def condition_to_device(condition: dict[str, Any], device: Any) -> dict[str, torch.Tensor]:
    return {k: _as_f32(v, device) for k, v in condition.items()}


def condition_mask_to_device(condition_mask: dict[str, Any], device: Any) -> dict[str, torch.Tensor]:
    return {k: torch.as_tensor(v, dtype=torch.bool, device=device) for k, v in condition_mask.items()}


def integrate_translation(
    vf: torch.nn.Module,
    source: Any,
    cond_t: dict[str, torch.Tensor] | None,
    *,
    condition_mask: dict[str, torch.Tensor] | None = None,
    is_genot: bool,
    state_dim: int,
    num_steps: int = 50,
    seed: int = 0,
    device: Any = None,
    return_trajectory: bool = False,
) -> torch.Tensor:
    from torchdiffeq import odeint

    if device is None:
        device = next(vf.parameters()).device
    src = _as_f32(source, device)

    if is_genot:
        # y0 = latent ~ N(0, I) in the target/generated space; the cells condition the field.
        gen = torch.Generator().manual_seed(int(seed))
        y0 = torch.randn(src.shape[0], int(state_dim), generator=gen).to(device=device)
        source_cells: torch.Tensor | None = src
    else:
        y0 = src
        source_cells = None

    t_grid = torch.linspace(0.0, 1.0, num_steps, device=device)

    def f(t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        t_exp = t.reshape(1, 1).expand(y.shape[0], 1)
        return vf(t_exp, y, cond_t, source_cells, condition_mask=condition_mask)

    with torch.no_grad():
        trajectory = odeint(f, y0, t_grid, method="euler")

    return trajectory if return_trajectory else trajectory[-1]


@register_predictor("ode")
class ODEPredictor(Predictor):

    def __init__(self, *, is_genot: bool, state_dim: int, num_steps: int = 50, seed: int = 0) -> None:
        self._is_genot = is_genot
        self._state_dim = int(state_dim)
        self._num_steps = int(num_steps)
        self._seed = int(seed)

    def predict(self, model: torch.nn.Module, batch: dict[str, Any]) -> torch.Tensor:
        device = next(model.parameters()).device
        cond = batch.get("condition")
        cond_t = condition_to_device(cond, device) if cond is not None else None
        mask = batch.get("condition_mask")
        condition_mask = condition_mask_to_device(mask, device) if mask is not None else None
        return integrate_translation(
            model,
            batch["source"],
            cond_t,
            condition_mask=condition_mask,
            is_genot=self._is_genot,
            state_dim=self._state_dim,
            num_steps=self._num_steps,
            seed=self._seed,
            device=device,
        )
