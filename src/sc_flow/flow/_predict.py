"""Shared inference: integrate the torch velocity field to translate cells.

The one ODE-integration used by **both** :meth:`~sc_flow.FlowMatching.predict` and the validation loop
(:class:`~sc_flow.core.training._harness.TrainingModule`), so a validation metric reflects
exactly what ``predict`` does. It is objective-agnostic apart from a single ``is_genot`` switch on the flow
endpoints (OTFM integrates the cells themselves; GENOT integrates from latent noise with the cell held as
the source-conditioning input).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from sc_flow.core.training._predictor import Predictor, register_predictor

__all__ = ["integrate_translation", "condition_to_device", "ODEPredictor"]


def _as_f32(v: Any, device: Any) -> torch.Tensor:
    """``float32`` torch tensor on ``device`` — accepts a numpy array OR a torch tensor already on a device.

    In the validation loop Lightning has already moved batch values onto the GPU, so ``np.asarray`` on them
    would raise ("can't convert cuda tensor to numpy"); handle the tensor case directly (no host round-trip).
    """
    if isinstance(v, torch.Tensor):
        return v.to(device=device, dtype=torch.float32)
    return torch.as_tensor(np.asarray(v, dtype=np.float32), device=device)


def condition_to_device(condition: dict[str, Any], device: Any) -> dict[str, torch.Tensor]:
    """Coerce a resolved condition dict to ``float32`` torch tensors on ``device`` (broadcast as-is)."""
    return {k: _as_f32(v, device) for k, v in condition.items()}


def integrate_translation(
    vf: torch.nn.Module,
    source: Any,
    cond_t: dict[str, torch.Tensor] | None,
    *,
    is_genot: bool,
    state_dim: int,
    num_steps: int = 50,
    seed: int = 0,
    device: Any = None,
    return_trajectory: bool = False,
) -> torch.Tensor:
    """Euler-integrate ``vf`` to translate ``source`` under ``cond_t`` — returns a tensor on ``device``.

    OTFM (``is_genot=False``): integrate the ODE from the cells ``source`` themselves (source → target),
    with no source-conditioning. GENOT (``is_genot=True``): integrate from latent noise ``~ N(0, I)`` in the
    target space (``state_dim``), with ``source`` held fixed as the velocity field's source input across the
    integration (noise → target | source) — *stochastic*, made reproducible by ``seed``. Runs under
    :func:`torch.no_grad`; the caller owns the model's train/eval mode and device placement.
    """
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
        return vf(t_exp, y, cond_t, source_cells)

    with torch.no_grad():
        trajectory = odeint(f, y0, t_grid, method="euler")

    return trajectory if return_trajectory else trajectory[-1]


@register_predictor("ode")
class ODEPredictor(Predictor):
    """Flow-matching inference: Euler-integrate the velocity field to translate a batch's ``source``.

    The one concrete :class:`~sc_flow.core.training._predictor.Predictor` for flow matching — wraps
    :func:`integrate_translation`, so the validation loop and the public ``FlowMatching.predict`` produce
    the same translation. OTFM integrates the cells themselves; GENOT integrates from latent noise with the
    cells held as the source-conditioning input (``is_genot``).
    """

    def __init__(self, *, is_genot: bool, state_dim: int, num_steps: int = 50, seed: int = 0) -> None:
        self._is_genot = is_genot
        self._state_dim = int(state_dim)
        self._num_steps = int(num_steps)
        self._seed = int(seed)

    def predict(self, model: torch.nn.Module, batch: dict[str, Any]) -> torch.Tensor:
        device = next(model.parameters()).device
        cond = batch.get("condition")
        cond_t = condition_to_device(cond, device) if cond is not None else None
        return integrate_translation(
            model,
            batch["source"],
            cond_t,
            is_genot=self._is_genot,
            state_dim=self._state_dim,
            num_steps=self._num_steps,
            seed=self._seed,
            device=device,
        )
