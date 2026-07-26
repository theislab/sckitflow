"""The **real** flow objectives (OTFM / GENOT) as :class:`sc_flow.Component`\\s — symmetric with the
contrastive objective, on live jax/ott code.

Wraps the runtime :class:`~sc_flow.flow._objectives.OTFMObjective` / ``GENOTObjective`` (torch + lazy jax/ott
coupling) in a portable :class:`sc_flow.training.ObjectiveConfig`: a **nested probability-path Component**
plus ``condition_mode`` / ``regularization`` / ``match_method`` / ``match_kwargs`` as fields. The two inputs
that are *not* part of a portable model spec — ``coupling_locs`` (data-derived) and ``seed`` (run-derived) —
are supplied at build time via :class:`FlowBuildContext`.

Escape hatch (proven here on real code): a portable ``match_kwargs`` (``epsilon``, ``scale_cost`` …)
round-trips; but a **live/custom OT cost function** placed in ``match_kwargs`` trains fine yet makes
``to_spec`` raise :class:`~sc_flow.PortabilityError` — a research object cannot be serialized into a spec.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from sc_flow._registry import Component, register_live
from sc_flow.training._config import ObjectiveConfig

__all__ = [
    "ProbabilityPath",
    "LinearGaussianPathConfig",
    "LinearDiracPathConfig",
    "FlowBuildContext",
    "LiveCostFn",
    "OTFMObjectiveConfig",
    "GENOTObjectiveConfig",
]


@dataclass
class FlowBuildContext:
    """Run/data inputs threaded into ``build`` — deliberately NOT part of the portable objective spec."""

    seed: int = 0
    coupling_locs: dict[str, str] = field(default_factory=dict)


# --- probability paths as Components (build the real BaseProbabilityPath runtimes) ----------------


class ProbabilityPath(Component):
    """Abstract family base for probability-path configs (unregistered). ``build(ctx) -> a runtime path``."""

    def build(self, context: Any = None) -> Any:
        raise NotImplementedError


@dataclass
class LinearGaussianPathConfig(ProbabilityPath, type_id="flow.path.linear_gaussian", version=1):
    sigma: float = 0.1

    def build(self, context: FlowBuildContext | None = None) -> Any:
        from sc_flow.flow.probability_paths._probability_paths import LinearGaussianProbabilityPath

        seed = 0 if context is None else context.seed
        return LinearGaussianProbabilityPath(sigma=self.sigma, prng=torch.Generator().manual_seed(seed))


@dataclass
class LinearDiracPathConfig(ProbabilityPath, type_id="flow.path.linear_dirac", version=1):
    sigma: float = 0.0

    def build(self, context: FlowBuildContext | None = None) -> Any:
        from sc_flow.flow.probability_paths._probability_paths import LinearDiracProbabilityPath

        return LinearDiracProbabilityPath(sigma=self.sigma)


# --- the runtime escape hatch: a live / custom OT cost function is not portable -------------------


@register_live
class LiveCostFn:
    """A pre-constructed / custom ``ott`` cost function. Usable at runtime (build), but a config that holds
    one has no portable form — ``to_spec`` raises ``PortabilityError``."""

    def __init__(self, fn: Any = None) -> None:
        self.fn = fn


# --- OTFM / GENOT objective configs (build the real runtimes) -------------------------------------


@dataclass
class _OTObjectiveConfig(ObjectiveConfig):
    """Shared fields for the OT-coupled objectives (unregistered intermediate base)."""

    probability_path: ProbabilityPath = field(default_factory=LinearGaussianPathConfig)  # nested Component
    condition_mode: str = "deterministic"
    regularization: float = 1.0
    match_method: str = "sinkhorn"  # "sinkhorn" | "unbalanced" | "gromov_wasserstein" | "independent"
    match_kwargs: dict[str, Any] = field(default_factory=dict)  # JSON round-trips; a LiveCostFn value raises

    def _build_runtime(self, cls: type, context: FlowBuildContext | None) -> Any:
        ctx = context or FlowBuildContext()
        return cls(
            self.probability_path.build(ctx),
            condition_mode=self.condition_mode,
            regularization=self.regularization,
            coupling_locs=dict(ctx.coupling_locs),
            match_method=self.match_method,
            match_kwargs=dict(self.match_kwargs) or None,
            seed=ctx.seed,
        )


@dataclass
class OTFMObjectiveConfig(_OTObjectiveConfig, type_id="flow.otfm", version=1):
    def build(self, context: FlowBuildContext | None = None) -> Any:
        from sc_flow.flow._objectives import OTFMObjective

        return self._build_runtime(OTFMObjective, context)


@dataclass
class GENOTObjectiveConfig(_OTObjectiveConfig, type_id="flow.genot", version=1):
    def build(self, context: FlowBuildContext | None = None) -> Any:
        from sc_flow.flow._objectives import GENOTObjective

        return self._build_runtime(GENOTObjective, context)
