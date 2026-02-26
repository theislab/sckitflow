import statistics
import time
from dataclasses import dataclass
from typing import Any

import diffrax as dfx
import jax
import jax.numpy as jnp
import pytest
import torch
from torch import Tensor

from sc_flow.backends.jax.solvers import ODESolver as JAXODESolver
from sc_flow.backends.torch.solvers import ODESolver as TorchODESolver
from sc_flow.backends.torch.solvers.wrapperjax.ode_solver import WrappedODESolver


@dataclass
class ConstantVFTorch:
    """CPU torch dynamics — used by TorchODESolver and WrappedODESolver."""

    a: torch.nn.Parameter

    def get_vf_fn(self, **vf_kwargs: Any):
        a = self.a

        def f(t: Tensor, y: Tensor) -> Tensor:
            return a * torch.ones_like(y)

        return f

    def zero_grad(self):
        if self.a.grad is not None:
            self.a.grad.zero_()


class ConstantVFJax:
    """Pure JAX dynamics — used by JAXODESolver."""

    def __init__(self, a: float = 0.3):
        self._a = a

    def get_vf_fn(self, **vf_kwargs):
        a = self._a

        def f(t, y, args=None):
            return jnp.ones_like(y) * a

        return f


def _time_block(fn, warmup: int = 1, repeats: int = 5) -> tuple[float, float]:
    """Run fn() warmup+repeats times, return (mean_ms, std_ms) over repeats."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return statistics.mean(times), statistics.stdev(times)


NUM_TIME_STEPS = 201
DIM = 64
T0, T1 = 0.0, 1.0


@pytest.fixture(scope="module")
def y_init_torch():
    return torch.randn(DIM)


@pytest.fixture(scope="module")
def time_grid():
    return torch.linspace(T0, T1, NUM_TIME_STEPS)


@pytest.fixture(scope="module")
def y_init_jax():
    return jnp.ones(DIM) * 0.5


def test_solver_timing_comparison(y_init_torch, y_init_jax, time_grid, capsys):
    """
    Compares wall-clock time for:
      1. JAX ODESolver          – forward only
      2. Torch ODESolver        – forward + backward
      3. Wrapped ODESolver      – forward + backward

    Prints a summary table; does not assert on timing (hardware-dependent).
    Only correctness assertions are made.
    """
    WARMUP = 2
    REPEATS = 5

    # ── 1. JAX solver (forward only) ─────────────────────────────────────────
    jax_dyn = ConstantVFJax(a=0.3)
    jax_solver = JAXODESolver(
        dynamics=jax_dyn,
        method=dfx.Euler(),
        device_id="cpu",
    )

    def jax_forward():
        result = jax_solver.solve(
            y_init_jax,
            T0,
            T1,
            num_time_steps=NUM_TIME_STEPS,
            return_trajectory=False,
        )
        jax.block_until_ready(result)

    jax_fwd_mean, jax_fwd_std = _time_block(jax_forward, WARMUP, REPEATS)

    # ── 2. Torch solver (forward + backward) ─────────────────────────────────
    def make_torch_dyn():
        return ConstantVFTorch(a=torch.nn.Parameter(torch.tensor(0.3)))

    torch_dyn = make_torch_dyn()
    torch_solver = TorchODESolver(dynamics=torch_dyn, method="euler", device_id="cpu")

    def torch_fwd_bwd():
        torch_dyn.zero_grad()
        y = y_init_torch.clone().requires_grad_(True)
        final = torch_solver.solve(
            source=y,
            time=time_grid,
            return_trajectory=False,
        )
        loss = final.sum()
        loss.backward()

    torch_mean, torch_std = _time_block(torch_fwd_bwd, WARMUP, REPEATS)

    # ── 3. Wrapped solver (forward + backward) ────────────────────────────────
    wrapped_dyn = make_torch_dyn()
    wrapped_solver = WrappedODESolver(dynamics=wrapped_dyn, method="euler", device_id="cpu")

    def wrapped_fwd_bwd():
        wrapped_dyn.zero_grad()
        y = y_init_torch.clone().requires_grad_(True)
        final = wrapped_solver.solve(
            source=y,
            t0=T0,
            t1=T1,
            num_time_steps=NUM_TIME_STEPS,
            solver_kwargs={},
            return_trajectory=False,
        )
        loss = final.sum()
        loss.backward()

    wrapped_mean, wrapped_std = _time_block(wrapped_fwd_bwd, WARMUP, REPEATS)

    # ── correctness checks ────────────────────────────────────────────────────
    # all solvers should agree on final value: y0 + a * (t1 - t0)
    torch_dyn2 = make_torch_dyn()
    torch_solver2 = TorchODESolver(dynamics=torch_dyn2, method="euler", device_id="cpu")
    torch_final = torch_solver2.solve(source=y_init_torch, time=time_grid, return_trajectory=False).detach()

    wrapped_dyn2 = make_torch_dyn()
    wrapped_solver2 = WrappedODESolver(dynamics=wrapped_dyn2, method="euler", device_id="cpu")
    wrapped_final = wrapped_solver2.solve(
        source=y_init_torch,
        t0=T0,
        t1=T1,
        num_time_steps=NUM_TIME_STEPS,
        solver_kwargs={},
        return_trajectory=False,
    )
    if hasattr(wrapped_final, "to"):
        wrapped_final = wrapped_final.to("cpu")

    assert torch.allclose(
        torch_final,
        torch.tensor(wrapped_final.numpy() if hasattr(wrapped_final, "numpy") else wrapped_final),
        atol=1e-2,
    ), "Wrapped and Torch solvers disagree on final state"

    # ── print timing table ────────────────────────────────────────────────────
    with capsys.disabled():
        print("\n")
        print("=" * 60)
        print(f"  Solver Timing Comparison  (dim={DIM}, steps={NUM_TIME_STEPS})")
        print("=" * 60)
        print(f"  {'Solver':<28} {'Mean (ms)':>10}  {'Std (ms)':>9}")
        print("-" * 60)
        print(f"  {'JAX (fwd only)':<28} {jax_fwd_mean:>10.2f}  {jax_fwd_std:>9.2f}")
        print(f"  {'Torch (fwd+bwd)':<28} {torch_mean:>10.2f}  {torch_std:>9.2f}")
        print(f"  {'Wrapped JAX (fwd+bwd)':<28} {wrapped_mean:>10.2f}  {wrapped_std:>9.2f}")
        print("=" * 60)
        print(f"  Wrapped overhead vs Torch: {wrapped_mean / torch_mean:.2f}x")
        print(f"  Wrapped overhead vs JAX:   {wrapped_mean / jax_fwd_mean:.2f}x")
        print("=" * 60)
