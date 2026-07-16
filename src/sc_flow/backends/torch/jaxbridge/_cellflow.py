"""CellFlow (OT flow-matching) trained by torch/Lightning over a JAX loss.

The numerics -- probability path, velocity-field evaluation, and the FM/encoder
loss -- are **CellFlow's own JAX code**, called here, not reimplemented. torch's
only jobs are to hold the parameters (:class:`torch.nn.Parameter`), run the
Lightning loop, and drive the optimizer; the JAX gradient reaches the optimizer
through :class:`~sc_flow.backends.torch.jaxbridge._bridge.JaxLossFunction`.

See :mod:`sc_flow.backends.torch.jaxbridge._bridge` for the ownership contract
(torch owns params; JAX mirrors them per step via DLPack).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import lightning.pytorch as pl
import torch

from sc_flow.backends.torch.jaxbridge._bridge import JaxLossFunction, torch_to_jax

__all__ = ["make_fm_value_and_grad", "CellFlowJaxModule"]


def make_fm_value_and_grad(
    vf: Any,
    probability_path: Any,
    condition_mode: str,
    regularization: float,
) -> Callable[..., tuple[jax.Array, Any]]:
    """Build the jitted ``value_and_grad`` of CellFlow's OT-FM loss.

    This mirrors the loss inside
    :meth:`cellflow.solvers.OTFlowMatching._get_vf_step_fn` **exactly**: it calls
    ``probability_path.compute_xt`` / ``compute_ut`` and ``vf.apply`` (CellFlow's
    JAX code) and only reassembles the four-line loss arithmetic. Nothing here is
    a torch reimplementation of the flow-matching math.

    Returns a callable ``(params, t, source, target, conditions, encoder_noise,
    rng) -> (loss, grads)`` where ``grads`` shares the pytree structure of
    ``params``.
    """
    apply_fn = vf.apply

    def loss_fn(
        params: Any,
        t: jnp.ndarray,
        source: jnp.ndarray,
        target: jnp.ndarray,
        conditions: dict[str, jnp.ndarray] | None,
        encoder_noise: jnp.ndarray,
        rng: jax.Array,
    ) -> jnp.ndarray:
        # --- verbatim from cellflow.solvers._otfm.OTFlowMatching loss_fn ---
        rng_flow, rng_encoder, rng_dropout = jax.random.split(rng, 3)
        x_t = probability_path.compute_xt(rng_flow, t, source, target)
        v_t, mean_cond, logvar_cond = apply_fn(
            {"params": params},
            t,
            x_t,
            conditions,
            encoder_noise=encoder_noise,
            rngs={"dropout": rng_dropout, "condition_encoder": rng_encoder},
        )
        u_t = probability_path.compute_ut(t, x_t, source, target)
        flow_matching_loss = jnp.mean((v_t - u_t) ** 2)
        condition_mean_regularization = 0.5 * jnp.mean(mean_cond**2)
        condition_var_regularization = -0.5 * jnp.mean(1 + logvar_cond - jnp.exp(logvar_cond))
        if condition_mode == "stochastic":
            encoder_loss = condition_mean_regularization + condition_var_regularization
        elif (condition_mode == "deterministic") and (regularization > 0):
            encoder_loss = condition_mean_regularization
        else:
            encoder_loss = 0.0
        return flow_matching_loss + encoder_loss
        # --- end verbatim ---

    return jax.jit(jax.value_and_grad(loss_fn))


class CellFlowJaxModule(pl.LightningModule):
    """LightningModule that trains CellFlow with JAX compute + a torch optimizer.

    Parameters
    ----------
    vf
        A :class:`cellflow.networks.ConditionalVelocityField` (its ``.apply`` and
        attributes carry all the JAX math).
    probability_path
        A CellFlow probability path (e.g. ``ConstantNoiseFlow``) exposing
        ``compute_xt`` / ``compute_ut``.
    params
        Initial flax parameter pytree (as produced by ``vf.init``/``create_train_state``).
        Copied into ``torch.nn.Parameter`` leaves -- torch then owns them.
    lr
        Adam learning rate.
    seed
        Seed for the per-step JAX rng stream (flow noise, dropout, encoder).

    The parameters become the module's single source of truth as a
    :class:`torch.nn.ParameterList`; JAX views them per step (see the bridge
    module). Only ``float32`` on a single device is supported by the DLPack path.
    """

    def __init__(
        self,
        vf: Any,
        probability_path: Any,
        params: Any,
        *,
        lr: float = 1e-4,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self._vf = vf
        self._probability_path = probability_path
        self._lr = lr

        leaves, self._treedef = jax.tree_util.tree_flatten(params)
        # torch owns the params: clone the mirrored view into an independent leaf.
        # Preserve the flax dtype (do NOT force float32) so the loss stays bit-exact
        # against a pure-JAX reference at the same precision -- e.g. float64 under
        # ``jax.config.update("jax_enable_x64", True)``. A silent downcast here would
        # make the torch loss diverge from a float64 JAX reference.
        from sc_flow.backends.torch.jaxbridge._bridge import jax_to_torch

        torch_leaves = [jax_to_torch(leaf).clone() for leaf in leaves]
        for lf in torch_leaves:
            if not lf.is_floating_point():
                raise TypeError(f"Parameter leaves must be floating-point for gradient descent; got {lf.dtype}.")
        self._params = torch.nn.ParameterList([torch.nn.Parameter(lf) for lf in torch_leaves])

        self._value_and_grad = make_fm_value_and_grad(
            vf, probability_path, vf.condition_mode, vf.regularization
        )
        self._key = jax.random.PRNGKey(seed)

    def jax_params(self) -> Any:
        """Reconstruct the flax parameter pytree from the torch source of truth.

        Zero-copy views over the live torch parameters -- use for inference/export.
        """
        leaves = [torch_to_jax(p) for p in self._params]
        return jax.tree_util.tree_unflatten(self._treedef, leaves)

    def _batch_to_jax(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        """Mirror a torch batch into JAX arrays (zero-copy).

        Expected keys: ``time`` (n, 1), ``source`` (n, d), ``target`` (n, d),
        ``encoder_noise`` (n, embedding_dim), and optionally ``conditions`` -- a
        dict of ``(n, max_combination_length, cond_dim)`` arrays already in JAX.
        """
        out: dict[str, Any] = {
            "time": torch_to_jax(batch["time"]),
            "source": torch_to_jax(batch["source"]),
            "target": torch_to_jax(batch["target"]),
            "encoder_noise": torch_to_jax(batch["encoder_noise"]),
        }
        conditions = batch.get("conditions")
        if conditions is not None:
            out["conditions"] = {
                k: (v if not isinstance(v, torch.Tensor) else torch_to_jax(v)) for k, v in conditions.items()
            }
        else:
            out["conditions"] = None
        return out

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        self._key, rng = jax.random.split(self._key)
        jb = self._batch_to_jax(batch)

        def value_and_grad_fn(params: Any) -> tuple[jax.Array, Any]:
            return self._value_and_grad(
                params, jb["time"], jb["source"], jb["target"], jb["conditions"], jb["encoder_noise"], rng
            )

        loss = JaxLossFunction.apply(value_and_grad_fn, self._treedef, *self._params)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self._lr)
