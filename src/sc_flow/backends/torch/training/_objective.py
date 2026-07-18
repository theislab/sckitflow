"""The training seam: a model (torch weights) + an objective (computes the loss).

The redesign turns "how a batch becomes a scalar loss" into a small, registerable
:class:`Objective`. The weights always live in a torch ``nn.Module`` (the *model*);
the objective decides *where the numerics run* — natively in torch, or in JAX via the
DLPack bridge with the torch weights mirrored per step. Both are trained by the one
:class:`~sc_flow.backends.torch.training._harness.SCFlowLightningModule`, so "torch
vs JAX compute" is a one-line objective swap rather than a second LightningModule.

A third party who installs the toolbox and wants a slightly different architecture
registers a model builder (and reuses an objective); a new training math registers an
objective (and reuses the harness + data plumbing).
"""

from __future__ import annotations

import abc
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import torch

__all__ = [
    "Objective",
    "register_objective",
    "register_architecture",
    "build_objective",
    "build_architecture",
    "OBJECTIVE_REGISTRY",
    "ARCHITECTURE_REGISTRY",
]


class Objective(abc.ABC):
    """Computes a scalar training loss for a model on a batch.

    The single seam the harness calls each step. Implementations own *where* the
    numerics run (torch or JAX) and how they read the batch; the harness only sees the
    returned loss (whose gradient must flow to ``model``'s torch parameters) and logs.
    """

    @abc.abstractmethod
    def compute_loss(self, model: torch.nn.Module, batch: Any) -> tuple[torch.Tensor, dict[str, Any]]:
        """Return ``(loss, logs)``. ``loss.backward()`` must reach ``model.parameters()``."""


# Registries so third-party architectures / objectives are discoverable by name.
# An architecture builder returns a torch ``nn.Module`` (the weights); an objective
# builder returns an :class:`Objective`.
ARCHITECTURE_REGISTRY: dict[str, Callable[..., torch.nn.Module]] = {}
OBJECTIVE_REGISTRY: dict[str, Callable[..., Objective]] = {}


def register_architecture(name: str) -> Callable[[Callable[..., torch.nn.Module]], Callable[..., torch.nn.Module]]:
    """Register a model (architecture) builder under ``name``.

    The builder returns a torch ``nn.Module`` holding the weights. Lets someone
    ``register_architecture("my-net")`` and select it by config without editing the
    framework.
    """

    def deco(builder: Callable[..., torch.nn.Module]) -> Callable[..., torch.nn.Module]:
        if name in ARCHITECTURE_REGISTRY:
            raise ValueError(f"Architecture {name!r} already registered.")
        ARCHITECTURE_REGISTRY[name] = builder
        return builder

    return deco


def register_objective(name: str) -> Callable[[Callable[..., Objective]], Callable[..., Objective]]:
    """Register an :class:`Objective` builder under ``name`` (e.g. ``"fm-linear"``, ``"otfm"``, ``"genot"``)."""

    def deco(builder: Callable[..., Objective]) -> Callable[..., Objective]:
        if name in OBJECTIVE_REGISTRY:
            raise ValueError(f"Objective {name!r} already registered.")
        OBJECTIVE_REGISTRY[name] = builder
        return builder

    return deco


def build_architecture(name: str, *args: Any, **kwargs: Any) -> torch.nn.Module:
    """Instantiate a registered architecture by name."""
    if name not in ARCHITECTURE_REGISTRY:
        raise KeyError(f"Architecture {name!r} not registered. Available: {sorted(ARCHITECTURE_REGISTRY)}.")
    return ARCHITECTURE_REGISTRY[name](*args, **kwargs)


def build_objective(name: str, *args: Any, **kwargs: Any) -> Objective:
    """Instantiate a registered objective by name."""
    if name not in OBJECTIVE_REGISTRY:
        raise KeyError(f"Objective {name!r} not registered. Available: {sorted(OBJECTIVE_REGISTRY)}.")
    return OBJECTIVE_REGISTRY[name](*args, **kwargs)


@register_objective("fm-linear")
class TorchLinearFMObjective(Objective):
    """Conditional flow-matching loss computed natively in torch.

    Straight-path CFM: for a batch of ``source``/``target`` (and optional ``cond``),
    sample ``t``, form ``x_t = (1-t) x0 + t x1`` and regress the model's velocity onto
    ``u = x1 - x0``. The model is called as ``model(t, x_t[, cond])``.
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


def _resolve_match_reps(batch: dict[str, Any], coupling_locs: Mapping[str, str] | None) -> tuple[np.ndarray, np.ndarray]:
    """The (source, target) reps to OT-match on: coupling reps when present, else the state reps."""
    if coupling_locs and "src_lin" in coupling_locs and "tgt_lin" in coupling_locs:
        return (
            np.asarray(batch["source_reps"][coupling_locs["src_lin"]], dtype=np.float32),
            np.asarray(batch["target_reps"][coupling_locs["tgt_lin"]], dtype=np.float32),
        )
    return np.asarray(batch["source"], dtype=np.float32), np.asarray(batch["target"], dtype=np.float32)


def _ot_indices(
    src_rep: np.ndarray,
    tgt_rep: np.ndarray,
    *,
    match_method: str,
    match_kwargs: Mapping[str, Any],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Minibatch linear OT plan -> resample indices (the one JAX call; forward-only, seeded)."""
    from sc_flow.backends.jax.coupling import independent_coupling, ot_linear_coupling

    if match_method == "independent":
        return independent_coupling(src_rep, tgt_rep, rng=rng)
    return ot_linear_coupling(src_rep, tgt_rep, method=match_method, rng=rng, **match_kwargs)


def _quadratic_indices(
    batch: dict[str, Any],
    coupling_locs: Mapping[str, str],
    match_kwargs: Mapping[str, Any],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Minibatch quadratic/fused Gromov-Wasserstein plan -> resample indices (JAX ott GW; seeded).

    Matches cells by **intra-domain structure** on the ``*_quad`` reps (fused with the ``*_lin`` reps when
    the schema provides both). Source/target quad reps may live in different spaces (GW is structural). The
    returned indices reorder the state ``source``/``target`` just like the linear path.
    """
    from sc_flow.backends.jax.coupling import ot_quadratic_coupling

    src_quad = np.asarray(batch["source_reps"][coupling_locs["src_quad"]], dtype=np.float32)
    tgt_quad = np.asarray(batch["target_reps"][coupling_locs["tgt_quad"]], dtype=np.float32)
    src_lin = tgt_lin = None
    if "src_lin" in coupling_locs and "tgt_lin" in coupling_locs:  # fused GW
        src_lin = np.asarray(batch["source_reps"][coupling_locs["src_lin"]], dtype=np.float32)
        tgt_lin = np.asarray(batch["target_reps"][coupling_locs["tgt_lin"]], dtype=np.float32)
    return ot_quadratic_coupling(
        src_quad, tgt_quad, source_lin=src_lin, target_lin=tgt_lin, rng=rng, **match_kwargs
    )


def _condition_tensors(
    cond: Mapping[str, Any] | None,
    tgt_ixs: np.ndarray,
    n_target: int,
    device: Any,
    dtype: Any,
) -> dict[str, torch.Tensor] | None:
    """Torch condition dict aligned to the coupled batch.

    A per-cell ``(Bt, …)`` condition follows the target reorder (§7.2); a leaf-level ``(1, mc, dim)``
    condition is left to broadcast.
    """
    if cond is None:
        return None
    out: dict[str, torch.Tensor] = {}
    for realm, arr in cond.items():
        a = np.asarray(arr, dtype=np.float32)
        if a.shape[0] == n_target:
            a = a[tgt_ixs]
        out[realm] = torch.as_tensor(a, device=device, dtype=dtype)
    return out


def _encoder_reg(
    mean: torch.Tensor | None, logvar: torch.Tensor | None, regularization: float
) -> torch.Tensor | None:
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


class _TorchOTObjective(Objective):
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
        ``{role: loc}`` from :attr:`CompiledData.coupling` (e.g. ``src_lin``/``tgt_lin``), or ``None``
        to OT-match on the state reps.
    match_method
        ``"sinkhorn"`` / ``"unbalanced"`` (OT via :func:`ot_linear_coupling`) or ``"independent"``.
    match_kwargs
        Extra kwargs forwarded to the coupling solver (e.g. ``epsilon``/``tau_a``/``tau_b``).
    seed
        Seed for this objective's stochastic sources — the OT plan-sampling
        (:class:`numpy.random.Generator`), the per-step ``t`` draw and (GENOT) the latent-noise draw
        (CPU :class:`torch.Generator` objects). Fixing it makes each step bit-reproducible (Sinkhorn itself
        is a deterministic solve), independent of process-global RNG state.
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
        # Seeded, explicit generators — coupling plan-sampling, the t draw, and (stochastic CE) the encoder
        # noise are reproducible regardless of global numpy/torch state. Generators stay on CPU.
        self._np_rng = np.random.default_rng(seed)
        self._t_gen = torch.Generator().manual_seed(self._seed)
        self._enc_gen = torch.Generator().manual_seed(self._seed + 2)

    def _encode(
        self, model: torch.nn.Module, cond_t: dict[str, torch.Tensor] | None
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Encode the condition **once** -> ``(embedding, mean, logvar)``.

        Deterministic encoder: ``embedding == mean``, ``logvar`` is ``None``. Stochastic encoder:
        ``embedding = mean + exp(0.5*logvar) * eps`` with seeded ``eps`` (reparameterization), and
        ``mean``/``logvar`` flow to the KL term. The single embedding is reused by the velocity field, so
        the reparameterization noise is consistent between the velocity and its regularization.
        """
        if cond_t is None or not getattr(model, "is_conditional", False):
            return None, None, None
        mean, logvar = model.condition_stats(cond_t)
        if logvar is None:
            return mean, mean, None
        eps = torch.randn(mean.shape, generator=self._enc_gen).to(device=mean.device, dtype=mean.dtype)
        return mean + torch.exp(0.5 * logvar) * eps, mean, logvar

    def _couple(self, batch: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        """OT-resample the batch: ``(src_ixs, tgt_ixs)`` on the coupling reps (seeded).

        Quadratic/fused-GW when the coupling schema is quadratic (``src_quad``/``tgt_quad``), else linear
        sinkhorn (or ``independent``). The generated state space is untouched — coupling only pairs cells.
        """
        if self._quad and self._match_method != "independent":
            return _quadratic_indices(batch, self._coupling_locs, self._match_kwargs, self._np_rng)
        src_rep, tgt_rep = _resolve_match_reps(batch, self._coupling_locs)
        return _ot_indices(
            src_rep, tgt_rep, match_method=self._match_method, match_kwargs=self._match_kwargs, rng=self._np_rng
        )


@register_objective("otfm")
class TorchOTFMObjective(_TorchOTObjective):
    """(OT) conditional flow-matching loss, computed in torch, coupling solved in JAX.

    Mirrors cellflow's ``OTFlowMatching`` step: each minibatch, resample the ``(source, target)`` pairing
    by a **minibatch OT plan** (the one JAX call — :func:`ot_linear_coupling`, forward-only, no gradient),
    then the straight-path CFM loss on the coupled pairs — ``x_t = compute_xt(t, source, target)``, regress
    ``model(t, x_t, cond)`` onto ``u = target - source`` — plus the deterministic encoder regularization.
    ``match_method="independent"`` gives cellflow's ``match_fn=None`` (vanilla CFM) baseline.
    """

    def compute_loss(self, model: torch.nn.Module, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        param = next(model.parameters())
        device, dtype = param.device, param.dtype

        src_state = np.asarray(batch["source"], dtype=np.float32)
        tgt_state = np.asarray(batch["target"], dtype=np.float32)
        src_ixs, tgt_ixs = self._couple(batch)

        x0 = torch.as_tensor(src_state[src_ixs], device=device, dtype=dtype)  # OTFM flows source -> target
        x1 = torch.as_tensor(tgt_state[tgt_ixs], device=device, dtype=dtype)
        cond_t = _condition_tensors(batch.get("condition"), tgt_ixs, tgt_state.shape[0], device, dtype)
        emb, mean, logvar = self._encode(model, cond_t)  # encode once (reparam if stochastic)

        # t drawn on CPU with the seeded generator (device-agnostic), then moved to the model's device.
        t = torch.rand(x0.shape[0], 1, generator=self._t_gen).to(device=device, dtype=dtype)
        x_t = self._path.compute_xt(t, x0, x1)
        u = self._path.compute_ut(t, x_t, x0, x1)
        v = model.velocity_from_embedding(t, x_t, emb)
        return _loss_with_reg(v, u, mean, logvar, self._regularization)


@register_objective("genot")
class TorchGENOTObjective(_TorchOTObjective):
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

        src_state = np.asarray(batch["source"], dtype=np.float32)
        tgt_state = np.asarray(batch["target"], dtype=np.float32)
        src_ixs, tgt_ixs = self._couple(batch)

        x0_source = torch.as_tensor(src_state[src_ixs], device=device, dtype=dtype)  # conditions the VF
        target = torch.as_tensor(tgt_state[tgt_ixs], device=device, dtype=dtype)
        cond_t = _condition_tensors(batch.get("condition"), tgt_ixs, tgt_state.shape[0], device, dtype)
        emb, mean, logvar = self._encode(model, cond_t)  # encode once (reparam if stochastic)

        # latent ~ N(0, I) in target space, drawn on CPU (seeded), then moved to the model's device.
        latent = torch.randn(target.shape, generator=self._latent_gen).to(device=device, dtype=dtype)
        t = torch.rand(target.shape[0], 1, generator=self._t_gen).to(device=device, dtype=dtype)
        x_t = self._path.compute_xt(t, latent, target)  # flow noise -> target
        u = self._path.compute_ut(t, x_t, latent, target)
        v = model.velocity_from_embedding(t, x_t, emb, source=x0_source)  # source-conditioned velocity field
        return _loss_with_reg(v, u, mean, logvar, self._regularization)
