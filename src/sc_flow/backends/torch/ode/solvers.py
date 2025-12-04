

from collections.abc import Callable
from typing import Any, Literal
from torch import device, Tensor, linspace, nn
from torchdiffeq import odeint 
from ..nn import BaseVelocityField, MLPUnconditionalVF

class ODESolver:
    "Base Class for ODE Solvers"
    def __init__(
            self,
            vf: BaseVelocityField = MLPUnconditionalVF(),
            num_time_steps: int = 500,
            solver_kwargs: dict[str, Any] | None = None,
            device_id: Literal["cuda", "cpu"] = "cuda",
    ) -> None:
        if solver_kwargs is None:
            solver_kwargs = {}
        solver_kwargs.setdefault("method", "euler")
        solver_kwargs.setdefault("atol", 1e-5)
        solver_kwargs.setdefault("rtol", 1e-5)
        
        self.vf = vf
        self.num_time_steps = num_time_steps
        self.solver = solver_kwargs
        self.device_id = device_id
        
        self.device = device(self.device_id)
        self.time = linspace(0.0, 1.0, self.num_time_steps).to(self.device)
        
    def solve(
        self, 
        source: Tensor,
        return_trajectory: bool = False,
    ) -> Tensor:
        trajectory = odeint(
            self.vf,
            source,
            self.time, 
            **self.solver_kwargs
        )
        return trajectory if return_trajectory else trajectory[-1]
        
        
        
