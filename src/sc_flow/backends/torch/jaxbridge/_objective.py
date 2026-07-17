"""A JAX-compute :class:`Objective` + a torch model that owns flax weights.

This adapts the DLPack bridge to the training seam so the *one*
:class:`~sc_flow.backends.torch.training._harness.SCFlowLightningModule` can train a
CellFlow flow-matching loss with all numerics in JAX while the optimizer steps torch
``nn.Parameter``s. ``JaxParamModule`` is the "model" (torch weights mirroring a flax
pytree); ``JaxFMObjective`` is the objective (JAX loss + gradient bridged into torch).
Swapping ``TorchLinearFMObjective`` for this is the whole "torch → JAX compute" move.
"""

from __future__ import annotations

from typing import Any

import jax
import torch

from sc_flow.backends.torch.jaxbridge._bridge import JaxLossFunction, jax_to_torch, torch_to_jax
from sc_flow.backends.torch.jaxbridge._cellflow import make_fm_value_and_grad
from sc_flow.backends.torch.training._objective import Objective, register_objective

__all__ = ["JaxParamModule", "CellFlowFMObjective"]


class JaxParamModule(torch.nn.Module):
    """The torch "model" whose parameters mirror a flax pytree (weights on torch).

    Flattens the flax params once and clones each leaf into a ``nn.Parameter`` in a
    deterministic order; ``treedef`` reassembles the pytree. This is the single source
    of truth the optimizer updates — JAX only ever sees a per-step DLPack view.
    """

    def __init__(self, params_pytree: Any) -> None:
        super().__init__()
        leaves, self.treedef = jax.tree_util.tree_flatten(params_pytree)
        self._leaves = torch.nn.ParameterList([torch.nn.Parameter(jax_to_torch(leaf).clone()) for leaf in leaves])

    @property
    def param_tensors(self) -> list[torch.nn.Parameter]:
        """Parameter leaves in ``treedef`` order (matches ``self.parameters()``)."""
        return list(self._leaves)


@register_objective("cellflow")
class CellFlowFMObjective(Objective):
    """CellFlow OT-FM loss computed in JAX, bridged into torch autograd.

    Built from a CellFlow velocity field + probability path (its numerics), it consumes
    a batch of ``time``/``source``/``target``/``encoder_noise``/``conditions`` tensors,
    mirrors them and the model's weights into JAX (zero-copy), runs
    ``jax.value_and_grad`` of the FM loss, and returns the loss through
    :class:`JaxLossFunction` so the torch optimizer updates the (torch) weights.

    The paired model must be a :class:`JaxParamModule` built from this velocity field's
    initial params.
    """

    def __init__(self, vf: Any, probability_path: Any, *, seed: int = 0) -> None:
        self._value_and_grad = make_fm_value_and_grad(vf, probability_path, vf.condition_mode, vf.regularization)
        self._key = jax.random.PRNGKey(seed)

    def compute_loss(self, model: torch.nn.Module, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        if not hasattr(model, "treedef") or not hasattr(model, "param_tensors"):
            raise TypeError("JaxFMObjective requires a JaxParamModule as the model.")
        self._key, rng = jax.random.split(self._key)

        time = torch_to_jax(batch["time"])
        source = torch_to_jax(batch["source"])
        target = torch_to_jax(batch["target"])
        encoder_noise = torch_to_jax(batch["encoder_noise"])
        conditions = batch.get("conditions")
        if conditions is not None:
            conditions = {k: (v if not isinstance(v, torch.Tensor) else torch_to_jax(v)) for k, v in conditions.items()}

        def value_and_grad_fn(params: Any) -> tuple[jax.Array, Any]:
            return self._value_and_grad(params, time, source, target, conditions, encoder_noise, rng)

        loss = JaxLossFunction.apply(value_and_grad_fn, model.treedef, *model.param_tensors)
        return loss, {"loss": loss.detach()}
