import torch
from torch import Tensor, nn
from sc_flow.backends.torch.nn import BaseVelocityField

class DummyVelocityField(BaseVelocityField):
    """
    Minimal concrete velocity field for testing ODESolver.

    Implements dx/dt = 3 t^2 (same as Meta's DummyModel), independent of x.
    Analytic solution over [0, 1]: x(1) = x(0) + 1.
    """

    def __init__(self) -> None:
        super().__init__()
        self._vf = self._make_vf()
        self.last_forward_kwargs: dict | None = None

    def _make_vf(self) -> nn.Module:
        # We don't need a real network here.
        return nn.ModuleDict()

    def forward(
        self,
        t: Tensor,
        x: Tensor,
        **kwargs,
    ) -> Tensor:
        
        self.last_forward_kwargs = kwargs
        
        return torch.ones_like(x) * 3.0 * t**2

    def get_vf_fn(self, **kwargs):
        def _vf_fn(t: Tensor, x: Tensor) -> Tensor:
            return self.forward(t, x, **kwargs)

        return _vf_fn

class ConstantVelocityField(BaseVelocityField):
    """
    Velocity field: dx/dt = a, where a is a learnable parameter.
    This is used to test gradient flow through the solver.
    """

    def __init__(self, init_a: float = 1.0) -> None:
        super().__init__()
        self.a = nn.Parameter(torch.tensor(init_a))
        self._vf = self._make_vf()

    def _make_vf(self) -> nn.Module:
        return nn.ModuleDict()

    def forward(
        self,
        t: Tensor,
        x: Tensor,
        **kwargs,
    ) -> Tensor:
        
        return torch.ones_like(x) * self.a

    def get_vf_fn(self, **kwargs):
        def _vf_fn(t: Tensor, x: Tensor) -> Tensor:
            return self.forward(t, x, **kwargs)

        return _vf_fn
