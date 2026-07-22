"""Generic torch/numpy type aliases for the ML-toolbox core.

Model-family-agnostic only: array/tensor/device aliases used across the core (and reused by the
flow-matching layer). Flow-matching-specific callables (velocity field, coupling/match, SDE drift, …) do
**not** live here — they belong to :mod:`sc_flow.flow` (e.g. ``VelocityFieldFn`` in ``flow/_vf.py``). Keeping
this file generic is what lets ``core`` be lifted into a standalone package.
"""

from collections.abc import Sequence

import numpy as np
import torch

try:
    from numpy.typing import NDArray

    NumpyArray = NDArray[np.float32 | np.float64]
except (ImportError, TypeError):
    NumpyArray = np.ndarray

ShapeLike = Sequence[int] | torch.Size
TensorLike = torch.Tensor | np.ndarray
MappedTensor = dict[str, torch.Tensor]
DeviceLike = str | torch.device
