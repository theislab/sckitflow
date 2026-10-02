"""The one place that turns a seed into generators."""

from __future__ import annotations

import numpy as np
import torch

__all__ = ["generators"]


def generators(seed: int, *key: int, device: torch.types.Device = None) -> tuple[torch.Generator, np.random.Generator]:
    """A fresh torch and numpy generator for the stream ``key`` of ``seed``, e.g. ``(stage, step)``.

    The same ``(seed, *key)`` always gives the same draws, so a resumed run repeats its streams without
    saved RNG state, and no global generator is read or written.
    """
    torch_seed, numpy_seed = np.random.SeedSequence(seed, spawn_key=key).generate_state(2, np.uint64)
    return torch.Generator(device=device).manual_seed(int(torch_seed)), np.random.default_rng(int(numpy_seed))
