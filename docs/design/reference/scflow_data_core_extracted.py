"""
sc_flow.data — CONTAINER-FREE CORE (extraction)
================================================

What this file is: sc_flow's data module with the *container part* deleted, so the
abstraction that remains is visible. Everything here is what sc_flow ALREADY has; the
only edits are (1) `DistributionData` -> the abstract `Distribution`, (2) inline notes
marking the exact seam where dagloader would plug in.

The container part that was removed (all of `sc_flow/data/containers/`):
    BaseData, StateData, CategoricalData, CouplingData, MixedTypeData, DistributionData
...plus the schemas that BUILD those containers from a single in-memory AnnData.

The key finding, up front
-------------------------
The tree (`NestedData`) and the sampler (`Sampler`) DO NOT depend on the containers.
They depend only on a 2-method protocol:

        Distribution:  __len__()  +  __getitem__(idx) -> Distribution

Everything container-specific (state/condition/groups/coupling arrays, `.X`, `.obsm`,
sort/concat, encoders) lives *below* that line. So "getting rid of the container part"
leaves a clean, container-agnostic skeleton. The catch for matching dagloader is a
SINGLE method — see `Sampler._sample_from_distr` and the NOTE blocks.
"""

from __future__ import annotations

import abc
from collections.abc import Callable, Iterable
from functools import cached_property, partial
from typing import Any, Generic, TypeVar

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────────────
# 1. THE ABSTRACTION (unchanged from sc_flow/data/_abc.py)
#    This is the whole "language": Distribution / MatchedDistributions / DataTree.
#    No bio, no container, no AnnData. Trees and leaves.
# ─────────────────────────────────────────────────────────────────────────────────────

DataT = TypeVar("DataT")
DistributionT = TypeVar("DistributionT", bound="Distribution")
MatchedDistributionsT = TypeVar("MatchedDistributionsT", bound="MatchedDistributions")
DataTreeT = TypeVar("DataTreeT", bound="DataTree")


class Distribution(abc.ABC):
    """A population you can measure the size of and draw a sub-population from.

    THIS IS THE ENTIRE SEAM. The container part (`DistributionData`) is just one
    implementation of these two methods over in-memory numpy. A dagloader-backed
    stream would be another implementation.
    """

    @abc.abstractmethod
    def __len__(self) -> int: ...

    @abc.abstractmethod
    def __getitem__(self, idx: np.ndarray | slice) -> "Distribution":
        """Return the sub-population at `idx`.

        NOTE (the one thing that blocks streaming): `idx` is an ARBITRARY random mask
        (see `Sampler._sample_indices`). Materialized containers do this in O(1) via
        numpy fancy-indexing. An out-of-core / streamed source CANNOT honor an arbitrary
        random mask — it can only yield the next weighted batch. See section 4.
        """
        ...


class MatchedDistributions(abc.ABC):
    """A leaf: a target population, optionally paired with a source population."""

    target: Distribution
    source: Distribution | None

    @abc.abstractmethod
    def align(self) -> "MatchedDistributions": ...


class DataTree(Generic[DataT], abc.ABC):
    """A tree whose leaves are `MatchedDistributions`."""

    @abc.abstractmethod
    def flatten(self) -> Iterable[MatchedDistributions]: ...


# ─────────────────────────────────────────────────────────────────────────────────────
# 2. THE LEAF (sc_flow/data/_composite.py::MatchedData, containers stripped)
#    Only `align()` ever looked "inside" a Distribution, and only for the continuous-
#    covariate row-count reconciliation. Everything else is len + slice.
# ─────────────────────────────────────────────────────────────────────────────────────


class MatchedData(MatchedDistributions):
    def __init__(self, target: Distribution, source: Distribution | None = None):
        self.target = target
        self.source = source
        # REMOVED: __post_init__ coupling-dim assertion — it reached into
        # target.target_coupling_data / source.source_coupling_data, i.e. into the
        # container. That check belongs to whatever the container guarantees, not here.

    @property
    def n_target_obs(self) -> int:
        return len(self.target)

    @property
    def n_source_obs(self) -> int | None:
        return None if self.source is None else len(self.source)

    def align(self) -> "MatchedData":
        # sc_flow's align() reconciles source vs target row counts by slicing/repeating
        # the SOURCE. It needs: len(source), source[slice], and concat(sources).
        # -> len + __getitem__ + a class-level concat. Still no container internals
        #    EXCEPT `target.has_continuous_condition_covariates` (a semantic flag).
        #
        # SEAM NOTE: `has_continuous_condition_covariates` is the ONLY container-semantic
        # bit the tree/leaf layer reads. To stay container-agnostic, promote it to the
        # Distribution protocol (a property) or move align() into the container layer.
        raise NotImplementedError("stays identical; shown only to mark its 3 requirements")


# ─────────────────────────────────────────────────────────────────────────────────────
# 3. THE TREE (sc_flow/data/_composite.py::NestedData, containers stripped)
#
#    THIS is where the user's point lives. Today the tree is built from ONE
#    `DistributionData` (`data`) by SLICING it: every leaf is `data[slice_for_group]`.
#    Source/target pairing is done by picking two slices of THE SAME array
#    (`source_key`, `matched_keys`). That is exactly what makes it:
#        - single-input   (one `data`)
#        - container-bound (slicing a materialized array)
#
#    dagloader inverts this: N independent sources, each its own Node (its own tree of
#    leaves), bound together. Multi-input and container-agnostic BY CONSTRUCTION.
# ─────────────────────────────────────────────────────────────────────────────────────


class NestedData(DataTree):
    """Recursive dict of tuple-key -> (subtree | MatchedData leaf)."""

    def __init__(self, mapping: dict):
        self.mapping = mapping

    def flatten(self) -> Iterable[MatchedData]:
        for v in self.mapping.values():
            if isinstance(v, NestedData):
                yield from v.flatten()
            else:
                yield v

    # -- how it's built today (the single-container assumption made explicit) ----------
    @classmethod
    def init_from_data(cls, data: Distribution, mapped_index, source_key=None, matched_keys=None):
        """
        `data`         : ONE Distribution (the whole sorted population).      <- single input
        `mapped_index` : tree of {group_key -> slice-into-`data`}.            <- random-access
        `source_key`   : one group is the shared source for all targets  (one-to-many)
        `matched_keys` : {source_key -> target_key} explicit pairs       (one-to-one)

        Every leaf = MatchedData(data[idx_target], data[idx_source]) — BOTH slices of the
        SAME `data`. Replace this single `data` + slice model with a per-leaf
        `(target_source, source_source)` pair and the tree becomes multi-input; that pair
        is precisely a dagloader (Node, bound-Node) — see section 4.
        """
        raise NotImplementedError("identical to sc_flow; shown to expose the single-`data` assumption")


# ─────────────────────────────────────────────────────────────────────────────────────
# 4. THE SAMPLER (sc_flow/data/samplers/_base.py) — ALREADY container-free.
#    It only ever calls: tree.flatten(), len(node.target), node.target[mask],
#    node.source[mask], node.__class__(t, source=s). Protocol-only. Verbatim below.
# ─────────────────────────────────────────────────────────────────────────────────────


class Sampler(Generic[MatchedDistributionsT, DataT], abc.ABC):
    def __init__(self, tree, *, use_nodes_weights=True, inverse_frequency_weights=True,
                 replace_samples=False, replace_nodes=False):
        self._tree = tree
        self._use_nodes_weights = use_nodes_weights
        self._inverse_frequency_weights = inverse_frequency_weights
        self._replace_samples = replace_samples
        self._replace_nodes = replace_nodes

    @abc.abstractmethod
    def _dispatch_node(self, node: MatchedDistributionsT) -> DataT: ...

    @cached_property
    def flattened_data(self):
        return tuple(self._tree.flatten())

    @cached_property
    def nodes_p(self) -> np.ndarray:
        # weight per leaf from TARGET obs counts (inverse-frequency by default).
        counts = np.array([len(e.target) for e in self.flattened_data])
        freq = counts / counts.sum()
        if self._inverse_frequency_weights:
            inv = 1 / freq
            return inv / inv.sum()
        return freq

    def _sample_nodes(self, n_nodes: int) -> np.ndarray:
        return np.random.choice(self.flattened_data, n_nodes,
                                p=self.nodes_p if self._use_nodes_weights else None,
                                replace=self._replace_nodes)

    def _sample_indices(self, n_obs: int, batch_size: int) -> np.ndarray:
        # ARBITRARY random mask over the leaf's rows — the streaming blocker.
        if self._replace_samples:
            return np.random.randint(0, n_obs, batch_size)
        return np.random.permutation(n_obs)[:batch_size]

    def _sample_from_distr(self, distr: Distribution, batch_size: int) -> Distribution:
        # ┌── THE ONE LINE THAT ASSUMES A MATERIALIZED, RANDOM-ACCESS CONTAINER ──┐
        mask = self._sample_indices(len(distr), batch_size)                      # │
        return distr[mask]                                                       # │
        # └───────────────────────────────────────────────────────────────────────┘
        # To unify with dagloader, this becomes ONE polymorphic call, e.g.
        #     return distr.draw(batch_size, rng)
        # materialized impl:  return self[self._rng_mask(batch_size)]   (today's behavior)
        # streamed impl:      return next(self._dagloader_stream)       (annbatch batch)

    def _sample_observations(self, node: MatchedDistributionsT, batch_size: int) -> DataT:
        target = self._sample_from_distr(node.target, batch_size)
        source = None if node.source is None else self._sample_from_distr(node.source, batch_size)
        return self._dispatch_node(node.__class__(target, source=source))

    def _sample(self, n_nodes: int, batch_size: int) -> tuple:
        nodes = self._sample_nodes(n_nodes)
        return tuple(map(partial(self._sample_observations, batch_size=batch_size), nodes))


class FSampler(Sampler):
    def __init__(self, tree, dispatch_fn: Callable | None = None, **kw):
        super().__init__(tree, **kw)
        self._dispatch_fn = (lambda x: x) if dispatch_fn is None else dispatch_fn

    def _dispatch_node(self, node):
        return self._dispatch_fn(node)


# ─────────────────────────────────────────────────────────────────────────────────────
# 5. WHERE dagloader MATCHES  (the mapping, as code comments)
# ─────────────────────────────────────────────────────────────────────────────────────
#
#   sc_flow (container-free core)          dagloader                     match quality
#   ─────────────────────────────────────  ────────────────────────────  ─────────────
#   DataTree / NestedData                  Scheme                        structural twin
#   MatchedData(target, source)            (root Node, bound child Node) exact
#   leaf = data[slice]                     Node leaf (weighted category) exact idea,
#                                                                        different backing
#   nodes_p (inverse-freq on target obs)   Node.weights on root          same knob*
#   source_key / matched_keys              Bind(parent, child, common)   exact (Bind
#                                                                        docstring cites
#                                                                        both by name)
#   ONE `data` sliced for every leaf       sources={name: Container},    <-- THE GAP:
#     (single-input, materialized)         each Node -> one source,      multi-input +
#                                          Container = AnnData |          container-agnostic
#                                          DatasetCollection | list       is native here
#   distr[arbitrary_mask]                  weighted batch STREAM          <-- THE OTHER GAP:
#     (random access)                      (ClassSampler, no rand access) random access is
#                                                                        not streamable
#
#   * sc_flow weights leaves by inverse target frequency at SAMPLE time; dagloader carries
#     explicit per-leaf weights on the Node. `uniform/frequency/inverse_frequency` in
#     dagloader._schema build exactly sc_flow's weighting as data.
#
# ─────────────────────────────────────────────────────────────────────────────────────
# 6. THE TWO CHANGES needed for sc_flow to "match" (become multi-input + streamable)
# ─────────────────────────────────────────────────────────────────────────────────────
#
#   (A) Multi-input: stop building the tree from ONE `DistributionData`. Let a leaf hold
#       target and source that come from DIFFERENT sources. In dagloader terms a leaf is
#       (root Node over source_A, bound child Node over source_B). This ALSO makes the
#       source↔target coupling a `Bind` (project on shared cols) instead of two slices of
#       one array — no code path in sections 2–4 changes, only `init_from_data`.
#
#   (B) Container-agnostic / streamable: widen `Distribution` from
#           {__len__, __getitem__(mask)}                       (random access)
#       to
#           {__len__?, draw(batch_size, rng) -> Distribution}  (batch pull)
#       Materialized `DistributionData` implements draw() as today's `self[mask]`.
#       A `StreamedDistribution` implements draw() as `next(dagloader_stream)`. The Sampler
#       calls `distr.draw(...)` instead of `distr[mask]` — the single edit in section 4.
#       (`__len__` is only needed for the weight vector; for a stream, weights come from
#        the Node instead, so `nodes_p` would read Node.weights rather than len(target).)
#
#   Net: the tree, the leaf, the weighting, and the source/target pairing are already a
#   1:1 match with dagloader. The container part is the ONLY thing that has to change, and
#   it changes behind the `Distribution` interface — exactly the layer this extraction
#   isolated.
