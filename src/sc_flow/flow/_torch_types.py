
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
