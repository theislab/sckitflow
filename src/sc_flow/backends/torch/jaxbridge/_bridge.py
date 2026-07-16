"""Zero-copy JAX <-> torch bridge for "torch-optimizes, JAX-computes".

The single mechanism in this file is :class:`JaxLossFunction`, a
:class:`torch.autograd.Function` whose *forward* runs a JAX loss and stashes the
JAX gradient, and whose *backward* hands that gradient back to torch autograd.
This is what lets a torch optimizer update parameters whose loss and gradient
were both produced by JAX -- **no flow-matching math is reimplemented in torch.**

Ownership
---------
**torch owns the parameters.** They live as :class:`torch.nn.Parameter` leaves
(the single source of truth). Each step they are *mirrored* into JAX as a pytree
of ``jax.Array`` leaves via DLPack -- a zero-copy view over the same device
buffer, not a copy. JAX computes ``loss`` and ``grad = d loss / d params``; the
gradient is mirrored back the same way and installed as ``param.grad`` by torch
autograd, so ``optimizer.step()`` updates the torch parameters in place. JAX
holds no persistent parameter state.

DLPack transfer
---------------
``torch_to_jax`` / ``jax_to_torch`` use the DLPack protocol
(``jax.dlpack.from_dlpack`` and ``torch.utils.dlpack.from_dlpack``) so the array
is shared, not serialized -- no host round-trip. Both frameworks must place the
buffer on the *same* device (e.g. both on ``cuda:0``); ``assert_same_device``
guards this. torch tensors that ``requires_grad`` cannot be exported through
DLPack, so ``torch_to_jax`` detaches first -- correctness is preserved because
the gradient path goes through :class:`JaxLossFunction`, not through the buffer
view.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import torch
from torch.utils import dlpack as torch_dlpack

__all__ = ["torch_to_jax", "jax_to_torch", "JaxLossFunction", "assert_same_device"]


def torch_to_jax(tensor: torch.Tensor) -> jax.Array:
    """View a torch tensor as a ``jax.Array`` (zero-copy, shared buffer).

    Detaches first: a grad-requiring tensor cannot be exported via DLPack, and
    the gradient path runs through :class:`JaxLossFunction` rather than this view.
    ``.contiguous()`` is required by the DLPack exporter for non-contiguous inputs.
    """
    return jax.dlpack.from_dlpack(tensor.detach().contiguous())


def jax_to_torch(array: jax.Array) -> torch.Tensor:
    """View a ``jax.Array`` as a torch tensor (zero-copy, shared buffer)."""
    return torch_dlpack.from_dlpack(array)


def assert_same_device(param_tensors: list[torch.Tensor]) -> None:
    """Fail loudly if parameters straddle devices JAX/torch cannot share.

    DLPack only shares a buffer when both frameworks address the same physical
    device. A silent host copy would defeat the whole point, so we check up front.
    """
    devices = {t.device for t in param_tensors}
    if len(devices) > 1:
        raise RuntimeError(f"All parameters must be on one device for the DLPack bridge; got {devices}.")


class JaxLossFunction(torch.autograd.Function):
    """Bridge a scalar JAX loss into torch autograd.

    ``forward`` receives the torch parameter leaves, mirrors them into a JAX
    pytree (via ``treedef``), and calls ``value_and_grad_fn(params) -> (loss,
    grads)`` -- a ``jax.value_and_grad`` closure that already carries the batch
    and rng. The per-parameter JAX gradients are stashed on ``ctx``; ``backward``
    returns ``grad_output * grad`` for each parameter (chain rule for a scalar
    loss), which torch accumulates into ``param.grad``.

    Because the value *and* the gradient are exactly what JAX computed, a torch
    ``loss.backward()`` reproduces the pure-JAX gradient to floating-point
    tolerance -- the bridge adds only DLPack views and a flatten/unflatten.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        value_and_grad_fn: Callable[[Any], tuple[jax.Array, Any]],
        treedef: Any,
        *param_tensors: torch.Tensor,
    ) -> torch.Tensor:
        assert_same_device(list(param_tensors))
        leaves = [torch_to_jax(p) for p in param_tensors]
        params = jax.tree_util.tree_unflatten(treedef, leaves)

        loss_jax, grads_jax = value_and_grad_fn(params)

        # grads share params' tree structure, so flattening with the same treedef
        # yields leaves aligned one-to-one with ``param_tensors``.
        grad_leaves = jax.tree_util.tree_leaves(grads_jax)
        ctx.grad_tensors = [jax_to_torch(g) for g in grad_leaves]

        # 0-d DLPack export is brittle across versions; reshape to (1,) for a
        # robust zero-copy view, then squeeze back to a scalar torch tensor.
        return jax_to_torch(loss_jax.reshape(1)).reshape(())

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[Any, ...]:  # type: ignore[override]
        # Two leading None slots for (value_and_grad_fn, treedef), which need no grad.
        grads = tuple(grad_output * g for g in ctx.grad_tensors)
        return (None, None, *grads)
