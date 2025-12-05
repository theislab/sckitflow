from collections.abc import Callable
from typing import Any, Literal
from torch import device, Tensor, linspace, nn
from torchdiffeq import odeint

from sc_flow.backends.torch.solvers.solver import Solver 
from ..nn import BaseVelocityField

class ODESolver(Solver):
    "Base Class for ODE Solvers"
    def __init__(
            self,
            vf: BaseVelocityField,
            num_time_steps: int = 500,
            atol: float = 1e-5,
            rtol: float = 1e-5,
            method: str = "euler",
            device_id: Literal["cuda", "cpu"] = "cuda",
    ) -> None:
        super().__init__()
        
        self.vf = vf
        self.num_time_steps = num_time_steps
        self.atol = atol
        self.rtol = rtol
        self.method = method
        
        self.device = device(device_id)
        self.time = linspace(0.0, 1.0, self.num_time_steps).to(self.device)
        
    def solve(
        self, 
        source: Tensor,
        return_trajectory: bool = False,
        options: dict[str, Any] | None = None, 
        **vf_kwargs: Any,
    ) -> Tensor:
        
        ode_func = self.vf.get_vf_fn(**vf_kwargs)
        source = source.to(self.device)
        
        trajectory = odeint(
            ode_func,
            source,
            self.time, 
            rtol=self.rtol,
            atol=self.atol,
            method=self.method,
            options=options,
        )
        return trajectory if return_trajectory else trajectory[-1]
        
        
        
