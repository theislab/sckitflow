from collections.abc import Callable
from typing import Any, Literal

import torchsde
from torch import Tensor, device
from torchsde import sdeint

from sc_flow.backends.torch.solvers.solver import Solver


class SDESolver(Solver):
    """Base Class for SDE Solvers methods"""

    def __init__(
        self,
        sde_type: Literal["ito", "stratonovich"] = "ito",
        noise_type: Literal["scalar", "diagonal", "general", "additive"] = "diagonal",
        method: str | None = None,
        device_id: Literal["cuda", "cpu"] = "cpu",
    ):
        super().__init__()
        self.method = method or "euler"
        self.sde_type = torchsde.SDEIto if sde_type == "ito" else torchsde.SDEStratonovich
        self.noise_type = noise_type
        self.device = device(device_id)

    def solve(
        self,
        source,
        time,
        drift_fn: Callable[[Tensor, Tensor], Tensor],
        diffusion_fn: Callable[[Tensor, Tensor], Tensor],
        *,
        rtol: float,
        atol: float,
        solver_kwargs: dict[str, Any],
    ) -> Any:
        """Integrate the SDE using a placeholder implementation."""
        source = source.to(self.device)
        time = time.to(self.device)

        _noise_type = self.noise_type
        _SDE_base = self.sde_type

        class _BaseSDE(_SDE_base):
            """SDE class for torchsde integration."""

            def __init__(self):
                super().__init__(noise_type=_noise_type)

            def f(self, t: Tensor, y: Tensor) -> Tensor:
                return drift_fn(t, y)

            def g(self, t: Tensor, y: Tensor) -> Tensor:
                return diffusion_fn(t, y)

        sde = _BaseSDE()

        trajectory = sdeint(
            sde,
            source,
            time,
            rtol=rtol,
            atol=atol,
            method=self.method,
            **solver_kwargs,
        )

        return trajectory
