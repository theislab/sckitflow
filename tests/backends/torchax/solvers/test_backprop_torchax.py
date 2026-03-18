"""
Minimal gradient debug tests for torchax JAX <-> torch interop.

Run with:
    pytest test_grad_debug.py -v -s

These tests progressively build up complexity, so when one fails
you know exactly which layer the gradient flow breaks at.
"""

from __future__ import annotations

import pytest
import torch
import torchax
from torchax.interop import j2t_autograd, jax_view, torch_view

torchax.enable_globally()


# ---------------------------------------------------------------------------
# Level 1: Pure torch autograd – baseline sanity check
# ---------------------------------------------------------------------------


def test_L1_pure_torch_baseline():
    """Vanilla torch autograd works. If this fails, environment is broken."""
    a = torch.tensor(2.0, requires_grad=True)
    y = a * 3.0  # dy/da = 3
    y.backward()
    assert a.grad is not None
    assert float(a.grad) == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Level 2: jax_view / torch_view round-trip preserves grad_fn
# ---------------------------------------------------------------------------


def test_L2_jax_view_roundtrip_preserves_requires_grad():
    """
    A torch tensor that requires grad should survive a jax_view -> torch_view
    round-trip and still be differentiable.
    """
    a = torch.tensor(2.0, requires_grad=True)
    a = a.to("jax")
    # Convert to JAX array and back
    a_jax = jax_view(a)
    a_back = torch_view(a_jax)

    # The round-tripped tensor should at least carry data correctly
    assert float(a_back) == pytest.approx(2.0), "value changed in round-trip"

    # NOTE: torchax tensors are _not_ autograd leaves after round-trip –
    # but they should still be usable in a j2t_autograd-wrapped function.
    # This test just checks the value; grad flow is tested in L3+.


# ---------------------------------------------------------------------------
# Level 3: j2t_autograd through a trivial JAX function (scalar * scalar)
# ---------------------------------------------------------------------------


def test_L3_j2t_autograd_scalar_multiply():
    """
    Wrap the simplest possible JAX function (multiply by constant) with
    j2t_autograd and check that gradients flow back to a torch Parameter.
    """

    def jax_fn(x):
        # x is a JAX array; multiply by 3
        return x * 3.0

    torch_fn = j2t_autograd(jax_fn)

    a = torch.tensor(2.0, requires_grad=True)
    a_torchax = a.to("jax")
    a_torchax.requires_grad_(True)
    out = torch_fn(a_torchax)  # back to torch

    out.backward()

    assert a_torchax.grad is not None, "grad is None – j2t_autograd did not propagate"
    assert float(a_torchax.grad) == pytest.approx(3.0), f"expected 3.0, got {a_torchax.grad}"


# ---------------------------------------------------------------------------
# Level 4: j2t_autograd through a simple vector field call  f(t, x) = a * x
# ---------------------------------------------------------------------------


def test_L4_j2t_autograd_vf_call():
    """
    Simulates what WrappedODESolver does for a single VF evaluation:
      - `a` is a torch Parameter
      - We wrap a JAX function that calls back into torch (via torch_view)
      - Gradients should flow back through j2t_autograd to `a`
    """

    a = torch.nn.Parameter(torch.tensor(0.3))

    # This mirrors what dynamics_wrapper / jax_vf does internally:
    # the JAX function receives JAX arrays, calls torch code via torch_view,
    # and returns a JAX array.
    def jax_vf(t, x, a_jax):
        a_torch = torch_view(a_jax)
        x_torch = torch_view(x)
        out_torch = a_torch * x_torch  # simple linear VF
        return jax_view(out_torch)

    torch_vf = j2t_autograd(jax_vf)

    t_jax = torch.tensor(0.5)
    t_torchax = t_jax.to("jax")
    x_jax = torch.tensor([1.0, 2.0, 3.0])
    x_torchax = x_jax.to("jax")
    a_torchax = a.to("jax")

    t_torchax.requires_grad_(False)  # time is not a parameter
    x_torchax.requires_grad_(False)  # state is not a parameter
    a_torchax.requires_grad_(True)  # this is the parameter we want gradients for

    out_jax = torch_vf(t_torchax, x_torchax, a_torchax)

    loss = out_jax.sum()  # sum(a * x) = a * 6, d/da = 6
    loss.backward()

    assert a_torchax.grad is not None, "grad is None – gradient did not reach Parameter `a`"
    assert float(a_torchax.grad) == pytest.approx(6.0, rel=1e-3), f"expected 6.0, got {float(a_torchax.grad)}"


# ---------------------------------------------------------------------------
# Level 5: gradient through a for-loop (mock Euler step)  -- closest to ODE
# ---------------------------------------------------------------------------


def test_L5_j2t_autograd_euler_loop():
    """
    Mock a single Euler integration step in JAX, wrapped with j2t_autograd,
    and check that gradients flow back to a torch Parameter `a`.

    ODE: dy/dt = a,  y(0) = y0,  t in [0, 1]
    Euler with N steps: y(1) = y0 + N * (1/N) * a = y0 + a
    d loss / d a  = d/da sum(y0 + a) = len(y0)
    """
    import jax

    a = torch.nn.Parameter(torch.tensor(0.5))
    y0 = torch.tensor([1.0, 0.0, -2.0])

    N = 10
    dt = 1.0 / N

    def jax_euler(y0_jax, a_jax):
        """Pure JAX Euler loop; a_jax is the scalar parameter."""
        a_jax = jax_view(a_jax)
        y0_jax = jax_view(y0_jax)

        def step(y, _):
            return y + dt * a_jax, None  # dy = a * dt at each step

        y_final, _ = jax.lax.scan(step, y0_jax, None, length=N)
        return y_final

    torch_euler = j2t_autograd(jax_euler)

    y0_jax = y0.to("jax")
    y0_jax.requires_grad_(False)  # initial state is not a parameter
    a_jax = a.to("jax")
    a_jax.requires_grad_(True)  # this is the parameter we want gradients for

    out_jax = torch_euler(y0_jax, a_jax)
    out = out_jax

    loss = out.sum()
    loss.backward()

    # d/da sum(y0 + a) = 3  (one contribution per element)
    assert a_jax.grad is not None, "grad is None – gradient did not survive jax.lax.scan"
    assert float(a_jax.grad) == pytest.approx(3.0, rel=1e-3, abs=1e-3), f"expected 3.0, got {float(a_jax.grad)}"
