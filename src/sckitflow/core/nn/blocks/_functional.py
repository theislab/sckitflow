from collections.abc import Callable

import torch

__all__ = ["FunctionalModule"]


class FunctionalModule(torch.nn.Module):
    def __init__(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> None:
        super().__init__()
        self._fn = fn
        # The identity module is required to register the parameters of modules
        # using this functional component; an error is thrown otherwise. Hence,
        # despite this being a no-op, we need to keep it nevertheless.
        self._net = torch.nn.Identity()

    def forward(self, x: torch.Tensor):
        return self._net(self._fn(x))
