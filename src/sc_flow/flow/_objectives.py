
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from sc_flow.training._objective import Objective, register_objective

__all__ = ["LinearFMObjective", "OTFMObjective", "GENOTObjective"]


@register_objective("fm-linear")
class LinearFMObjective(Objective):

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
    """Move a STATE-stream array to ``device`` as float32 (source/target are always float model inputs)."""
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=torch.float32)
    return torch.as_tensor(np.asarray(x, dtype=np.float32), device=device)


def _to_condition(x: Any, device: Any, float_dtype: Any) -> torch.Tensor:
    """Move a CONDITION-stream array to ``device`` preserving its kind: an integer **index** (categorical
    realm) becomes ``long`` for embedding/one-hot; a float **feature** vector is aligned to the model dtype.
    The realm's dtype *is* the contract — no realm-kind branching, just ``is_floating_point()``.
    """
    t = x.to(device=device) if isinstance(x, torch.Tensor) else torch.as_tensor(np.asarray(x), device=device)
    return t.to(float_dtype) if t.is_floating_point() else t.to(torch.long)


def _condition_tensors(
    cond: Mapping[str, Any] | None,
    tgt_ixs: torch.Tensor,
    n_target: int,
    device: Any,
    dtype: Any,
) -> dict[str, torch.Tensor] | None:
    if cond is None:
        return None
    out: dict[str, torch.Tensor] = {}
    for realm, arr in cond.items():
        a = _to_condition(arr, device, dtype)
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
