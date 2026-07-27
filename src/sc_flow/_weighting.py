"""Per-leaf sampling weights — which condition a batch is drawn from (scfit normalizes; missing leaf = 0)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from sc_flow._component import ComponentRegistry, ComponentSpec

__all__ = ["UNIFORM", "WEIGHTING_REGISTRY", "WeightingContext", "WeightingSpec", "build_weighting"]

WeightingSpec = ComponentSpec
UNIFORM: WeightingSpec = {"type": "sc_flow.uniform", "version": 1, "config": {}}


@dataclass(frozen=True)
class WeightingContext:
    leaves: Sequence[tuple]
    group_by: tuple[str, ...]


WEIGHTING_REGISTRY: ComponentRegistry[WeightingContext] = ComponentRegistry("weighting")


def _key(leaf: Any) -> tuple[str, ...]:
    """A leaf tuple or a ``"val1|val2"`` config key → a str tuple (leaf values arrive as numpy scalars)."""
    return tuple(leaf.split("|")) if isinstance(leaf, str) else tuple(str(x) for x in leaf)


@WEIGHTING_REGISTRY.register("sc_flow.uniform")
@dataclass(frozen=True)
class UniformConfig:
    """Every condition equally likely."""

    def build(self, context: WeightingContext) -> dict[tuple, float]:
        return dict.fromkeys(context.leaves, 1.0)


@WEIGHTING_REGISTRY.register("sc_flow.group_uniform")
@dataclass(frozen=True)
class GroupUniformConfig:
    """Equal total mass per ``by`` group (e.g. ``by: [dataset]``), uniform within it."""

    by: Sequence[str] = ()

    def build(self, context: WeightingContext) -> dict[tuple, float]:
        idx = [context.group_by.index(c) for c in self.by]
        groups: dict[tuple, list[tuple]] = {}
        for lf in context.leaves:
            groups.setdefault(tuple(str(lf[i]) for i in idx), []).append(lf)
        return {lf: 1.0 / (len(groups) * len(members)) for members in groups.values() for lf in members}


@WEIGHTING_REGISTRY.register("sc_flow.custom")
@dataclass(frozen=True)
class CustomConfig:
    """Explicit ``{"val1|val2": weight}`` over every leaf; 0 excludes a condition."""

    weights: Mapping[Any, float] = field(default_factory=dict)

    def build(self, context: WeightingContext) -> dict[tuple, float]:
        w = {_key(k): float(v) for k, v in self.weights.items()}
        return {lf: w[_key(lf)] for lf in context.leaves}


def build_weighting(
    spec: WeightingSpec | None, *, leaves: Sequence[tuple], group_by: Sequence[str]
) -> dict[tuple, float]:
    """``spec`` of :obj:`None` (or empty) means uniform. Validated by the registry."""
    return WEIGHTING_REGISTRY.build(spec or UNIFORM, WeightingContext(leaves=leaves, group_by=tuple(group_by)))
