"""Portable configs for the couplings a `Matcher` pairs a batch with.

Each config only binds its fields to a coupling function, which delegates the solve to POT. Which
solvers exist, what they accept and the couplings they return therefore depend on the installed
POT version as well.

There is one config per solver, holding only the fields that solver reads, so a setting it would
ignore fails validation as an unknown field.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import ClassVar

from scfit.registry import Component, component

from sckitflow.core._types import MatchFn, MatchFnOut, ScaleMethod
from sckitflow.core.coupling._coupling import independent_coupling, ot_linear_coupling, ot_quadratic_coupling

__all__ = [
    "CouplingConfig",
    "IndependentCouplingConfig",
    "ExactCouplingConfig",
    "SinkhornCouplingConfig",
    "PartialCouplingConfig",
    "UnbalancedCouplingConfig",
    "GromovWassersteinCouplingConfig",
    "FusedGromovWassersteinCouplingConfig",
]


class CouplingConfig(Component):
    """Family base for the couplings. ``build`` binds ``method`` and every field to ``fn`` as keyword arguments.

    Field defaults must equal ``fn``'s, which the tests check.
    """

    fn: ClassVar[Callable[..., MatchFnOut]]
    method: ClassVar[str | None] = None

    def build(self) -> MatchFn:
        kwargs = dict(self)
        if self.method is not None:
            kwargs["method"] = self.method
        return partial(type(self).fn, **kwargs)


@component("coupling.independent")
class IndependentCouplingConfig(CouplingConfig):
    """Random pairs."""

    fn = independent_coupling


class _LinearCouplingConfig(CouplingConfig):
    """Linear optimal transport, see `ot_linear_coupling` for the fields."""

    fn = ot_linear_coupling

    scale_cost: ScaleMethod = "mean"
    num_iter_max: int | None = None


@component("coupling.pot.exact")
class ExactCouplingConfig(_LinearCouplingConfig):
    """Exact linear optimal transport, POT's ``emd``."""

    method = "exact"


class _EntropicCouplingConfig(_LinearCouplingConfig):
    reg: float = 0.5
    stop_thr: float | None = None


@component("coupling.pot.sinkhorn")
class SinkhornCouplingConfig(_EntropicCouplingConfig):
    """Entropic linear optimal transport, POT's ``sinkhorn``."""

    method = "sinkhorn"


@component("coupling.pot.partial")
class PartialCouplingConfig(_EntropicCouplingConfig):
    """Entropic partial linear optimal transport, POT's ``entropic_partial_wasserstein``."""

    method = "partial"

    m: float | None = None


@component("coupling.pot.unbalanced")
class UnbalancedCouplingConfig(_EntropicCouplingConfig):
    """Entropic unbalanced linear optimal transport, POT's ``sinkhorn_knopp_unbalanced``."""

    method = "unbalanced"

    reg_m: float = 1.0


class _QuadraticCouplingConfig(CouplingConfig):
    """Quadratic optimal transport, see `ot_quadratic_coupling` for the fields."""

    fn = ot_quadratic_coupling

    # Scaled like the linear couplings: unscaled squared distances make the entropic solver underflow to an
    # all-zero plan on float32 data.
    scale_cost: ScaleMethod = "mean"
    epsilon: float = 1.0
    max_iter: int = 1000
    tol: float = 1e-9


@component("coupling.pot.gromov_wasserstein")
class GromovWassersteinCouplingConfig(_QuadraticCouplingConfig):
    """Entropic Gromov-Wasserstein, POT's ``entropic_gromov_wasserstein``."""

    method = "entropic_gromov_wasserstein"


@component("coupling.pot.fused_gromov_wasserstein")
class FusedGromovWassersteinCouplingConfig(_QuadraticCouplingConfig):
    """Entropic fused Gromov-Wasserstein, POT's ``entropic_fused_gromov_wasserstein``."""

    method = "entropic_fused_gromov_wasserstein"

    alpha: float = 0.5
