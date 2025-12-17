from typing import Any, Literal

from torch import Tensor, device
from torchdiffeq import odeint

from sc_flow.backends.torch.nn._vf import BaseVelocityField
from sc_flow.backends.torch.solvers.solver import Solver


class ODESolver(Solver):
    """Base Class for ODE Solvers"""

    def __init__(
        self,
        method: str = "euler",
        device_id: Literal["cuda", "cpu"] = "cpu",
    ) -> None:
        super().__init__()
        self.method = method
        self.device = device(device_id)

    def solve(
        self,
        vf: BaseVelocityField,
        source: Tensor,
        time: Tensor,
        *,
        rtol: float,
        atol: float,
        vf_kwargs: dict[str, Any] | None = None,
        solver_kwargs: dict[str, Any],
    ) -> Tensor:
        """Integrate the ODE using torchdiffeq's odeint."""
        if solver_kwargs is None:
            solver_kwargs = {}

        if vf_kwargs is None:
            vf_kwargs = {}

        diff_eqn = vf.get_vf_fn(**vf_kwargs)

        source = source.to(self.device)
        time = time.to(self.device)

        trajectory = odeint(
            diff_eqn,
            source,
            time,
            rtol=rtol,
            atol=atol,
            method=self.method,
            **solver_kwargs,
        )
        return trajectory
