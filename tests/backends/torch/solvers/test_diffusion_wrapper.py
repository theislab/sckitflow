# tests/backends/torch/solvers/test_diffusion_wrapper.py


import pytest
import torch
import torchsde
from torch import Tensor

from sc_flow.backends.torch.solvers.sde_solver import SDESolver


def _wrap(diffusion_fn, df_kwargs: dict | None = None):
    """Use SDESolver._get_diffusion_fn_wrapper without calling __init__."""
    solver = SDESolver.__new__(SDESolver)
    return solver._get_diffusion_fn_wrapper(diffusion_fn, df_kwargs=df_kwargs)


def test_wrapper_uses_t_only_when_one_param_after_partial():
    calls = []

    def g_raw(t: Tensor, *, sigma: float):
        calls.append(("g_raw", t.clone(), sigma))
        return sigma * torch.ones(3)

    wrapped = _wrap(g_raw, df_kwargs={"sigma": 0.5})

    t = torch.tensor(0.1)
    y = torch.zeros(3)
    out = wrapped(t, y, args=None)

    assert out.shape == (3,)
    assert torch.allclose(out, 0.5 * torch.ones(3))

    assert len(calls) == 1
    name, t_called, sigma_called = calls[0]
    assert name == "g_raw"
    assert torch.allclose(t_called, t)
    assert sigma_called == 0.5


def test_wrapper_uses_t_y_when_two_params_after_partial():
    calls = []

    def g_raw(t: Tensor, y: Tensor, *, sigma: float):
        calls.append(("g_raw", t.clone(), y.clone(), sigma))
        return sigma * torch.ones_like(y)

    wrapped = _wrap(g_raw, df_kwargs={"sigma": 0.3})

    t = torch.tensor(0.2)
    y = torch.arange(5.0)
    out = wrapped(t, y, args=None)

    assert out.shape == y.shape
    assert torch.allclose(out, 0.3 * torch.ones_like(y))

    assert len(calls) == 1
    name, t_called, y_called, sigma_called = calls[0]
    assert name == "g_raw"
    assert torch.allclose(t_called, t)
    assert torch.allclose(y_called, y)
    assert sigma_called == 0.3


def test_wrapper_no_df_kwargs_g_t_y():
    calls = []

    def g_raw(t: Tensor, y: Tensor):
        calls.append(("g_raw", t.clone(), y.clone()))
        return torch.ones_like(y)

    wrapped = _wrap(g_raw, df_kwargs=None)

    t = torch.tensor(0.0)
    y = torch.arange(3.0)

    out = wrapped(t, y, args=None)

    assert out.shape == y.shape
    assert torch.allclose(out, torch.ones_like(y))

    assert len(calls) == 1
    name, t_called, y_called = calls[0]
    assert name == "g_raw"
    assert torch.allclose(t_called, t)
    assert torch.allclose(y_called, y)


def test_wrapper_no_df_kwargs_g_t_only():
    calls = []

    def g_raw(t: Tensor):
        calls.append(("g_raw", t.clone()))
        return torch.full((4,), 7.0)

    wrapped = _wrap(g_raw, df_kwargs=None)

    t = torch.tensor(1.23)
    y = torch.zeros(4)

    out = wrapped(t, y, args=None)

    assert out.shape == (4,)
    assert torch.allclose(out, torch.full((4,), 7.0))

    assert len(calls) == 1
    name, t_called = calls[0]
    assert name == "g_raw"
    assert torch.allclose(t_called, t)


class _SDEWithWrapper(torchsde.SDEIto):
    """Simple SDE using a wrapped diffusion function."""

    def __init__(self, wrapped_diffusion):
        super().__init__(noise_type="diagonal")
        self._wrapped_diffusion = wrapped_diffusion

    def f(self, t: Tensor, y: Tensor) -> Tensor:
        # zero drift
        return torch.zeros_like(y)

    def g(self, t: Tensor, y: Tensor) -> Tensor:
        # TorchSDE in this version doesn't pass args, so we call with args=None
        return self._wrapped_diffusion(t, y, args=None)


@pytest.mark.parametrize("dim", [1, 4])
def test_sdeint_with_wrapped_diffusion_t_y(dim: int):
    """Check that torchsde.sdeint runs end-to-end for g(t, y, *, sigma)."""

    def g_raw(t: Tensor, y: Tensor, *, sigma: float):
        return sigma * torch.ones_like(y)

    wrapped = _wrap(g_raw, df_kwargs={"sigma": 0.5})
    sde = _SDEWithWrapper(wrapped)

    batch_size = 1
    y0 = torch.zeros(batch_size, dim)  # (batch, channels)
    ts = torch.linspace(0.0, 1.0, 11)

    sol = torchsde.sdeint(
        sde,
        y0,
        ts,
        dt=1e-2,
        rtol=1e-5,
        atol=1e-5,
        method="euler",
    )

    assert sol.shape == (len(ts), batch_size, dim)
    assert torch.all(torch.isfinite(sol))


@pytest.mark.parametrize("dim", [2, 5])
def test_sdeint_with_wrapped_diffusion_t_only(dim: int):
    """Integration test for a diffusion that only depends on t (and df_kwargs)."""

    def g_raw(t: Tensor, *, sigma: float):
        # We'll broadcast to match y via torch.ones_like in the wrapper call.
        # Here we return a 1D tensor; wrapped_diffusion will still call g(t).
        return sigma * torch.ones(dim)

    wrapped = _wrap(g_raw, df_kwargs={"sigma": 0.2})

    class _SDE(torchsde.SDEIto):
        def __init__(self):
            super().__init__(noise_type="diagonal")

        def f(self, t: Tensor, y: Tensor) -> Tensor:
            return torch.zeros_like(y)

        def g(self, t: Tensor, y: Tensor) -> Tensor:
            # g_raw(t) returns (dim,), but torchsde with diagonal noise expects (batch, dim)
            # so we expand to batch dimension here.
            g_t = wrapped(t, y, args=None)  # shape (dim,)
            return g_t.expand_as(y)  # (batch, dim)

    sde = _SDE()

    batch_size = 1
    y0 = torch.zeros(batch_size, dim)
    ts = torch.linspace(0.0, 0.5, 6)

    sol = torchsde.sdeint(
        sde,
        y0,
        ts,
        dt=1e-2,
        rtol=1e-5,
        atol=1e-5,
        method="euler",
    )

    assert sol.shape == (len(ts), batch_size, dim)
    assert torch.all(torch.isfinite(sol))
