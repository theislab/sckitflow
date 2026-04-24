from dataclasses import dataclass
from typing import Any

import torch

__all__ = ["StepData"]


@dataclass
class StepData:
    target_state: torch.Tensor
    target_coupling_lin: torch.Tensor | None
    target_coupling_quad: torch.Tensor | None
    target_condition_data: Any | None
    target_group_data: Any | None
    source_state: torch.Tensor | None
    source_coupling_lin: torch.Tensor | None
    source_coupling_quad: torch.Tensor | None
    source_condition_data: Any | None
    source_group_data: Any | None
