"""Concrete flow-matching objectives (the FM loss math), registered on import.

The generic seam (:class:`~sc_flow.core.training._objective.Objective` + registries) lives in the ML
core; these are the flow-matching implementations. The weights always live in a torch ``nn.Module``; the
OT-coupled objectives solve the minibatch transport plan in JAX via the DLPack bridge, imported **lazily**
(:func:`sc_flow._optional.require`) only when a coupling actually runs — so ``import sc_flow.flow`` and the
``match_method="independent"`` path need no jax. Importing this module registers ``fm-linear``, ``otfm``
and ``genot``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from sc_flow.core.training._objective import Objective, register_objective

__all__ = ["LinearFMObjective", "OTFMObjective", "GENOTObjective"]


@register_objective("fm-linear")
class LinearFMObjective(Objective):
    """Conditional flow-matching loss computed natively in torch.

    Straight-path CFM: for a batch of ``source``/``target`` (and optional ``cond``), sample ``t``, form
    ``x_t = (1-t) x0 + t x1`` and regress the model's velocity onto ``u = x1 - x0``. The model is called
    as ``model(t, x_t[, cond])``.
    """

    def compute_loss(self, model: torch.nn.Module, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        x0, x1 = batch["source"], batch["target"]
        cond = batch.get("cond")
        t = torch.rand(x0.shape[0], 1, device=x0.device, dtype=x0.dtype)
        x_t = (1.0 - t) * x0 + t * x1
        u = x1 - x0
        v = model(t, x_t) if cond is None else model(t, x_t, cond)
        loss = ((v - u) ** 2).mean()
        return loss, {"loss": loss.detach()}


# --- shared OT-coupled flow-matching plumbing (used by "otfm" and "genot") -----------------------


def _to_device(x: Any, device: Any) -> torch.Tensor:
    """Coerce a batch value to a ``float32`` torch tensor on ``device``.

    Handles a binded numpy array or a torch tensor Lightning already moved to the GPU; the batch stays
    where the model lives (no CPU round-trip).
    """
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=torch.float32)
    return torch.as_tensor(np.asarray(x, dtype=np.float32), device=device)


def _condition_tensors(
    cond: Mapping[str, Any] | None,
    tgt_ixs: torch.Tensor,
    n_target: int,
    device: Any,
    dtype: Any,
) -> dict[str, torch.Tensor] | None:
    """Torch condition dict aligned to the coupled batch, on ``device``.

    A per-cell ``(Bt, …)`` condition follows the target reorder (§7.2) via ``tgt_ixs``; a leaf-level
    ``(1, mc, dim)`` condition is left to broadcast.
    """
    if cond is None:
        return None
    out: dict[str, torch.Tensor] = {}
    for realm, arr in cond.items():
        a = _to_device(arr, device).to(dtype)
        if a.shape[0] == n_target:
            a = a[tgt_ixs]
        out[realm] = a
    return out


def _condition_masks(
    masks: Mapping[str, Any] | None,
    tgt_ixs: torch.Tensor,
    n_target: int,
    device: Any,
) -> dict[str, torch.Tensor] | None:
    """Boolean condition masks aligned with the same target reorder as condition tensors."""
    if masks is None:
        return None
    out: dict[str, torch.Tensor] = {}
    for realm, arr in masks.items():
        mask = torch.as_tensor(arr, dtype=torch.bool, device=device)
        if mask.shape[0] == n_target:
            mask = mask[tgt_ixs]
        out[realm] = mask
    return out


def _encoder_reg(mean: torch.Tensor | None, logvar: torch.Tensor | None, regularization: float) -> torch.Tensor | None:
    """Condition-encoder regularization (cellflow's) from the precomputed encoder stats.

    Deterministic (``logvar is None``): L2 ``0.5 * mean(mean**2)``, gated by ``regularization > 0``.
    Stochastic: the VAE KL ``0.5 * mean(mean**2 + exp(logvar) - logvar - 1)`` to ``N(0, I)`` (always on,
    matching cellflow's stochastic ``encoder_loss``).
    """
    if mean is None:
        return None
    if logvar is not None:
        return 0.5 * (mean**2 + torch.exp(logvar) - logvar - 1.0).mean()
    if regularization > 0:
        return 0.5 * (mean**2).mean()
    return None


def _loss_with_reg(
    v: torch.Tensor,
    u: torch.Tensor,
    mean: torch.Tensor | None,
    logvar: torch.Tensor | None,
    regularization: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """FM MSE ``mean((v - u)**2)`` plus the encoder regularization (see :func:`_encoder_reg`)."""
    fm_loss = ((v - u) ** 2).mean()
    loss = fm_loss
    logs: dict[str, Any] = {"fm_loss": fm_loss.detach()}
    reg = _encoder_reg(mean, logvar, regularization)
    if reg is not None:
        loss = fm_loss + reg
        logs["encoder_reg"] = reg.detach()
    logs["loss"] = loss.detach()
    return loss, logs


class _OTObjective(Objective):
    """Shared setup for OT-coupled torch flow-matching objectives (``otfm``, ``genot``).

    Holds the torch probability path, the coupling configuration, and the seeded generators; subclasses
    implement :meth:`compute_loss` and differ only in the **flow endpoints** and whether the (resampled)
    source cell **conditions** the velocity field.

    Parameters
    ----------
    probability_path
        A torch probability path exposing ``compute_xt(t, x0, x1)`` / ``compute_ut(t, xt, x0, x1)``.
    condition_mode
        ``"deterministic"`` (only mode the torch encoder supports). ``"stochastic"`` raises.
    regularization
        Gate for the encoder regularization term (cellflow semantics: ``> 0`` includes it, unscaled).
    coupling_locs
        ``{role: loc}`` from :attr:`CompiledData.coupling` (e.g. ``src_lin``/``tgt_lin``), or ``None`` to
        OT-match on the state reps.
    match_method
        ``"sinkhorn"`` / ``"unbalanced"`` (OT, solved in JAX) or ``"independent"`` (a torch random pairing,
        needs no jax).
    match_kwargs
        Extra kwargs forwarded to the coupling solver (e.g. ``epsilon``/``tau_a``/``tau_b``).
    seed
        Seed for this objective's stochastic sources — the OT plan-sampling (:class:`numpy.random.Generator`),
        the per-step ``t`` draw and (GENOT) the latent-noise draw (CPU :class:`torch.Generator` objects).
        Fixing it makes each step bit-reproducible (Sinkhorn itself is a deterministic solve), independent of
        process-global RNG state.
    """

    def __init__(
        self,
        probability_path: Any,
        *,
        condition_mode: str = "deterministic",
        regularization: float = 1.0,
        coupling_locs: Mapping[str, str] | None = None,
        match_method: str = "sinkhorn",
        match_kwargs: Mapping[str, Any] | None = None,
        seed: int = 0,
    ) -> None:
        self._path = probability_path
        self._condition_mode = condition_mode
        self._regularization = regularization
        self._coupling_locs = dict(coupling_locs) if coupling_locs else None
        self._match_method = match_method
        self._match_kwargs = dict(match_kwargs) if match_kwargs else {}
        # Quadratic/GW coupling when the schema names quad reps (fused if it also names lin reps).
        self._quad = bool(self._coupling_locs) and {"src_quad", "tgt_quad"} <= set(self._coupling_locs)
        self._seed = int(seed)
        # Seeded generators — the t draw, the encoder-noise draw (stochastic CE), the latent draw (GENOT),
        # the independent-coupling permutations, and the OT plan-sampling (a JAX PRNGKey) are all
        # reproducible regardless of global RNG state. torch generators stay on CPU (device-agnostic); the
        # jax key is created lazily on first use so importing the objective needs no jax.
        self._t_gen = torch.Generator().manual_seed(self._seed)
        self._enc_gen = torch.Generator().manual_seed(self._seed + 2)
        self._perm_gen = torch.Generator().manual_seed(self._seed + 3)
        self._coupling_key: Any = None

    def _encode(
        self,
        model: torch.nn.Module,
        cond_t: dict[str, torch.Tensor] | None,
        cond_mask: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Encode the condition **once** -> ``(embedding, mean, logvar)``.

        Deterministic encoder: ``embedding == mean``, ``logvar`` is ``None``. Stochastic encoder:
        ``embedding = mean + exp(0.5*logvar) * eps`` with seeded ``eps`` (reparameterization), and
        ``mean``/``logvar`` flow to the KL term. The single embedding is reused by the velocity field, so
        the reparameterization noise is consistent between the velocity and its regularization.
        """
        if cond_t is None or not getattr(model, "is_conditional", False):
            return None, None, None
        if cond_mask is None:
            mean, logvar = model.condition_stats(cond_t)
        else:
            mean, logvar = model.condition_stats(cond_t, condition_mask=cond_mask)
        if logvar is None:
            return mean, mean, None
        eps = torch.randn(mean.shape, generator=self._enc_gen).to(device=mean.device, dtype=mean.dtype)
        return mean + torch.exp(0.5 * logvar) * eps, mean, logvar

    def _couple(self, batch: dict[str, Any], device: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """OT-resample the batch on ``device`` → ``(src_ixs, tgt_ixs)`` torch long tensors.

        Quadratic/fused-GW when the schema is quadratic (``src_quad``/``tgt_quad``), else linear sinkhorn
        (or ``independent`` — a torch random pairing, no jax). The reps **and** the transport plan stay on
        ``device`` (GPU under CUDA) via zero-copy DLPack; only the tiny index arrays are produced. The
        generated state space is untouched — coupling only pairs cells. JAX is imported lazily here (and
        only here), so the ``independent`` path and plain ``import sc_flow.flow`` never need it.
        """
        locs = self._coupling_locs
        if self._match_method == "independent":
            n_src = _to_device(batch["source"], device).shape[0]
            n_tgt = _to_device(batch["target"], device).shape[0]
            m = min(n_src, n_tgt)
            src_ixs = torch.randperm(n_src, generator=self._perm_gen)[:m]
            tgt_ixs = torch.randperm(n_tgt, generator=self._perm_gen)[:m]
            return src_ixs.to(device), tgt_ixs.to(device)

        from sc_flow._optional import require

        jax = require("jax")
        couple_device = require("sc_flow.flow.coupling._device").couple_device

        if self._coupling_key is None:
            self._coupling_key = jax.random.PRNGKey(self._seed)
        self._coupling_key, sub = jax.random.split(self._coupling_key)

        if self._quad:
            src_rep = _to_device(batch["source_reps"][locs["src_quad"]], device)
            tgt_rep = _to_device(batch["target_reps"][locs["tgt_quad"]], device)
            src_lin = _to_device(batch["source_reps"][locs["src_lin"]], device) if "src_lin" in locs else None
            tgt_lin = _to_device(batch["target_reps"][locs["tgt_lin"]], device) if "tgt_lin" in locs else None
            return couple_device(
                src_rep, tgt_rep, key=sub, quad=True, src_lin=src_lin, tgt_lin=tgt_lin, match_kwargs=self._match_kwargs
            )

        if locs and "src_lin" in locs and "tgt_lin" in locs:
            src_rep = _to_device(batch["source_reps"][locs["src_lin"]], device)
            tgt_rep = _to_device(batch["target_reps"][locs["tgt_lin"]], device)
        else:
            src_rep = _to_device(batch["source"], device)
            tgt_rep = _to_device(batch["target"], device)
        return couple_device(src_rep, tgt_rep, key=sub, match_kwargs=self._match_kwargs)


@register_objective("otfm")
class OTFMObjective(_OTObjective):
    """(OT) conditional flow-matching loss, computed in torch, coupling solved in JAX.

    Mirrors cellflow's ``OTFlowMatching`` step: each minibatch, resample the ``(source, target)`` pairing by
    a **minibatch OT plan** (the one JAX call, forward-only, no gradient), then the straight-path CFM loss on
    the coupled pairs — ``x_t = compute_xt(t, source, target)``, regress ``model(t, x_t, cond)`` onto
    ``u = target - source`` — plus the deterministic encoder regularization. ``match_method="independent"``
    gives cellflow's ``match_fn=None`` (vanilla CFM) baseline and needs no jax.
    """

    def compute_loss(self, model: torch.nn.Module, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        param = next(model.parameters())
        device, dtype = param.device, param.dtype

        src_ixs, tgt_ixs = self._couple(batch, device)  # torch long indices on device
        source = _to_device(batch["source"], device)
        target = _to_device(batch["target"], device)
        x0 = source[src_ixs].to(dtype)  # OTFM flows source -> target
        x1 = target[tgt_ixs].to(dtype)
        cond_t = _condition_tensors(batch.get("condition"), tgt_ixs, target.shape[0], device, dtype)
        cond_mask = _condition_masks(batch.get("condition_mask"), tgt_ixs, target.shape[0], device)
        emb, mean, logvar = self._encode(model, cond_t, cond_mask)  # encode once (reparam if stochastic)

        # t drawn on CPU with the seeded generator (device-agnostic), then moved to the model's device.
        t = torch.rand(x0.shape[0], 1, generator=self._t_gen).to(device=device, dtype=dtype)
        x_t = self._path.compute_xt(t, x0, x1)
        u = self._path.compute_ut(t, x_t, x0, x1)
        v = model.velocity_from_embedding(t, x_t, emb)
        return _loss_with_reg(v, u, mean, logvar, self._regularization)


@register_objective("genot")
class GENOTObjective(_OTObjective):
    """GENOT (generative entropic OT) loss, computed in torch, coupling solved in JAX.

    Mirrors cellflow's ``GENOT`` step: OT-resample ``(source, target)`` (as OTFM), then sample a latent
    noise in **target space** and flow **latent → target**, while the (resampled) **source cell conditions**
    the velocity field (``model(t, x_t, cond, source=x0)``) — the source is *not* the flow's start.
    Endpoints are ``(latent, target)``: ``x_t = compute_xt(t, latent, target)``, ``u = target - latent``.

    Requires the velocity field to be built **with a source encoder** (``FlowMatching`` does this when
    ``objective="genot"``). Same coupling/condition/reg plumbing and seeded reproducibility as OTFM; adds a
    distinct seeded generator for the latent noise. Linear coupling only (GENOT-L); quadratic/GW is G2.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Distinct seeded stream for the target-space latent noise (CPU, device-agnostic).
        self._latent_gen = torch.Generator().manual_seed(self._seed + 1)

    def compute_loss(self, model: torch.nn.Module, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        param = next(model.parameters())
        device, dtype = param.device, param.dtype

        src_ixs, tgt_ixs = self._couple(batch, device)  # torch long indices on device
        source = _to_device(batch["source"], device)
        target = _to_device(batch["target"], device)
        x0_source = source[src_ixs].to(dtype)  # conditions the VF
        target_r = target[tgt_ixs].to(dtype)
        cond_t = _condition_tensors(batch.get("condition"), tgt_ixs, target.shape[0], device, dtype)
        cond_mask = _condition_masks(batch.get("condition_mask"), tgt_ixs, target.shape[0], device)
        emb, mean, logvar = self._encode(model, cond_t, cond_mask)  # encode once (reparam if stochastic)

        # latent ~ N(0, I) in target space, drawn on CPU (seeded), then moved to the model's device.
        latent = torch.randn(target_r.shape, generator=self._latent_gen).to(device=device, dtype=dtype)
        t = torch.rand(target_r.shape[0], 1, generator=self._t_gen).to(device=device, dtype=dtype)
        x_t = self._path.compute_xt(t, latent, target_r)  # flow noise -> target
        u = self._path.compute_ut(t, x_t, latent, target_r)
        v = model.velocity_from_embedding(t, x_t, emb, source=x0_source)  # source-conditioned velocity field
        return _loss_with_reg(v, u, mean, logvar, self._regularization)
