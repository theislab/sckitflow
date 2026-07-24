"""sc_flow-side data-layer glue: the few streaming helpers ``scfit.data`` does not expose.

``scfit.data`` is the absorbed streaming layer (the former ``binded``), but its public surface is a
subset: it exposes the schema (:class:`~scfit.data.Scheme`/:class:`~scfit.data.Node`/
:class:`~scfit.data.Bind`), the training :class:`~scfit.data.Loader`, ``split_scheme`` and the low-level
IO primitives (``open_source``/``obs_columns``/``materialize_node``/``get_from_container``) — but **not**
``uniform``, a ``key_backings`` alias, the ``ConditionLookup`` type, nor an eval-time loader. Rather than
reach back into the old ``binded`` package (removed) or edit the ``scfit`` base package, sc-flow-tools
derives those here, on top of ``scfit.data``'s public primitives.

The centrepiece is :class:`EvalLoader` — a small, **in-memory** control→perturbed eval adaptor. It reuses
``scfit.data.materialize_node`` to read each held-out node's cells into RAM once (a held-out split is
bounded), then yields, per held-out condition, the ``{"source", "target", "condition", "leaf"}`` batch the
perturbation-validation callback scores. It is deliberately isolated from the training data path and speaks
sc-flow-tools' own ``condition_lookup`` contract (the **full** leaf over the node's ``cols``).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from scfit.data import Scheme
    from scfit.data._schema import SamplerConfig

__all__ = ["ConditionLookup", "uniform", "key_backings", "EvalLoader", "make_train_loader"]

#: A per-leaf condition resolver: maps a leaf tuple (over a node's ``cols``) to ``{realm: array}`` — a
#: categorical realm's integer index or a feature realm's looked-up vector. Produced by ``compile_obs``.
ConditionLookup = Callable[[tuple], dict[str, np.ndarray]]


def uniform(combos: Any) -> dict[tuple, float]:
    """Every combination equally likely — the ``{combo: weight}`` mapping a :class:`~scfit.data.Node` takes.

    Mirrors the former ``binded.uniform``; ``scfit.data`` defines it internally but does not export it.
    """
    return {tuple(c): 1.0 for c in combos}


def key_backings(source: Any, loc: str) -> list:
    """The array(s) backing rep ``loc`` for a source (one per store), each exposing ``.shape``.

    A thin, stable alias over ``scfit.data``'s ``get_from_container`` (the former ``binded.key_backings``,
    renamed on absorption). ``compile_obs`` uses ``key_backings(source, loc)[0].shape[1]`` to size a rep
    without materializing cells.
    """
    from scfit.data._io import get_from_container

    return get_from_container(source, loc)


def _state_matrix(adata: Any, loc: str) -> np.ndarray:
    """Dense rows of rep ``loc`` (``"X"`` | ``"obsm/<k>"`` | ``"layers/<k>"``) from an in-memory AnnData."""
    if loc == "X":
        x = adata.X
    else:
        field, sub = loc.split("/", 1)
        x = getattr(adata, field)[sub]
    dense = getattr(x, "todense", None)
    return np.asarray(dense() if dense is not None else x)


class EvalLoader:
    """In-memory control→perturbed eval adaptor over a held-out :class:`~scfit.data.Scheme`.

    Reads the two nodes (root = perturbed/target, its one bound child = control/source) into RAM once via
    :func:`scfit.data.materialize_node`, groups controls by the bind's ``common`` (match-context) columns,
    and yields one batch per held-out condition: ``{"source", "target", "condition", "leaf"}`` where
    ``source``/``target`` are float32 torch tensors and ``condition`` is ``condition_fn(leaf)``.

    Isolated from the training loader on purpose (see the module docstring) and matched to sc-flow-tools'
    ``condition_lookup``: it passes the **full** leaf (over the node's ``cols``), not a context-stripped one.

    Parameters
    ----------
    scheme
        Held-out :class:`~scfit.data.Scheme` — root perturbed node + exactly one bound control child.
    sampler_config
        Only ``batch_size`` is read: the cap on target cells emitted per condition (controls are emitted
        in full; the validation callback caps them downstream).
    condition_fn
        Maps a perturbed leaf to its condition payload (the compiled ``condition_lookup``). ``None`` omits
        ``"condition"`` from the batch.
    seed
        Seed for sampling target cells down to ``batch_size`` (reproducible across ``iter_conditions``).
    """

    def __init__(
        self,
        scheme: Scheme,
        sampler_config: SamplerConfig,
        condition_fn: ConditionLookup | None = None,
        *,
        seed: int = 0,
    ) -> None:
        from scfit.data._io import materialize_node

        binds = [b for b in scheme.binds if b.parent == scheme.root]
        if len(binds) != 1:
            raise ValueError("EvalLoader expects exactly one bound source of the root.")
        bind = binds[0]
        pert = scheme.nodes[scheme.root]  # perturbed / target
        ctrl = scheme.nodes[bind.child]  # control / source
        common = bind.common
        self._cfg = sampler_config
        self._cond_fn = condition_fn
        self._seed = int(seed)

        # Materialize each node's positive-weight cells once (held-out split is bounded). The state rep is
        # the first key (compile_obs puts the sample_rep first; any extra keys are coupling reps eval ignores).
        adata_p, codes_p, leaves_p = materialize_node(scheme.sources[pert.source], pert)
        adata_c, codes_c, leaves_c = materialize_node(scheme.sources[ctrl.source], ctrl)
        state_p = _state_matrix(adata_p, pert.keys[0])
        state_c = _state_matrix(adata_c, ctrl.keys[0])

        # Group controls by their match-context (the bind's shared columns), so each perturbed condition
        # draws its source from the same context — the "control = same group" matching, done in memory.
        p_ctx = [pert.cols.index(c) for c in common]
        c_ctx = [ctrl.cols.index(c) for c in common]
        ctrl_rows: dict[tuple, list[int]] = {}
        for i, code in enumerate(codes_c):
            ctx = tuple(str(leaves_c[code][p]) for p in c_ctx)
            ctrl_rows.setdefault(ctx, []).append(i)
        self._ctrl_by_ctx = {ctx: state_c[np.asarray(rows, dtype=np.int64)] for ctx, rows in ctrl_rows.items()}

        # One condition per perturbed leaf that has a matching control population.
        pert_rows: dict[int, list[int]] = {}
        for i, code in enumerate(codes_p):
            pert_rows.setdefault(code, []).append(i)
        self._conditions: list[tuple[tuple, np.ndarray, tuple]] = []
        skipped = 0
        for code, rows in pert_rows.items():
            leaf = tuple(leaves_p[code])
            ctx = tuple(str(leaf[p]) for p in p_ctx)
            if ctx not in self._ctrl_by_ctx:
                skipped += 1
                continue
            self._conditions.append((leaf, state_p[np.asarray(rows, dtype=np.int64)], ctx))
        if not self._conditions:
            raise ValueError("no held-out condition has a matching control population to evaluate.")
        self._skipped = skipped

    @property
    def n_conditions(self) -> int:
        return len(self._conditions)

    def iter_conditions(self, n_conditions: int | None = None) -> Iterator[dict]:
        """Yield one batch per held-out condition; with ``n_conditions`` set, cycle conditions to that many."""
        import torch

        conds = self._conditions
        if n_conditions is None:
            order = list(range(len(conds)))
        else:
            reps = int(np.ceil(n_conditions / len(conds)))
            order = (list(range(len(conds))) * reps)[:n_conditions]

        rng = np.random.default_rng(self._seed)
        batch_size = self._cfg.batch_size
        for j in order:
            leaf, target, ctx = conds[j]
            source = self._ctrl_by_ctx[ctx]
            if target.shape[0] > batch_size:  # bound target cells per condition; controls stay full
                target = target[rng.choice(target.shape[0], size=batch_size, replace=False)]
            out: dict[str, Any] = {
                "leaf": leaf,
                "source": torch.as_tensor(np.ascontiguousarray(source), dtype=torch.float32),
                "target": torch.as_tensor(np.ascontiguousarray(target), dtype=torch.float32),
            }
            if self._cond_fn is not None:
                out["condition"] = self._cond_fn(leaf)
            yield out


class _SemanticBatches:
    """Adapt ``scfit.data.Loader``'s node-keyed batches to sc-flow-tools' semantic training contract.

    ``scfit.data.Loader`` emits ``{root_node: {loc: rows}, child_node: {loc: rows}, "annotations": {...}}``
    (the former ``binded`` emitted ``source``/``target``/``condition`` directly). The flow objectives read
    ``batch["source"]``/``["target"]``/``["condition"]`` (+ ``["source_reps"]``/``["target_reps"]`` for the
    coupling reps), so this remaps each batch: the root (perturbed) node → ``target``, its bound child
    (control) node → ``source``, the state rep (each node's first key) → the main matrix, and the per-leaf
    ``annotations[root]`` → ``condition``.
    """

    def __init__(self, loader: Any, root: str, child: str, state_key: str) -> None:
        self._loader = loader
        self._root = root
        self._child = child
        self._state_key = state_key

    def __iter__(self) -> Iterator[dict]:
        for node_batch in self._loader:
            pert = node_batch[self._root]  # {loc: rows} — the target (perturbed) node's aligned reps
            ctrl = node_batch[self._child]  # {loc: rows} — the source (control) node's aligned reps
            out: dict[str, Any] = {
                "source": ctrl[self._state_key],
                "target": pert[self._state_key],
                "source_reps": ctrl,  # keyed by rep loc — the coupling objectives index their coupling locs
                "target_reps": pert,
            }
            annotations = node_batch.get("annotations")
            if annotations is not None and self._root in annotations:
                out["condition"] = annotations[self._root]  # {realm: array} for the batch's (single) leaf
            yield out


def make_train_loader(
    scheme: Scheme,
    sampler_config: SamplerConfig,
    condition_lookup: ConditionLookup,
    *,
    preload_to_gpu: bool = False,
) -> _SemanticBatches:
    """Build a ``scfit.data.Loader`` for training and adapt it to sc-flow-tools' semantic batch contract.

    Precomputes the root (perturbed) node's per-leaf conditions into the ``annotations`` mapping
    ``scfit.data.Loader`` now takes (``{node: {leaf: {realm: array}}}``), attaches ``preload_to_gpu`` (a
    ``Loader`` argument, no longer a ``SamplerConfig`` field), and wraps the node-keyed batches via
    :class:`_SemanticBatches`. The state rep is each node's first key (``compile_obs`` puts ``sample_rep``
    first). Isolated here so ``scfit.data``'s loader/batch shape stays out of the model facade.
    """
    from scfit.data import Loader

    root = scheme.root
    children = [b.child for b in scheme.binds if b.parent == root]
    if len(children) != 1:
        raise ValueError("make_train_loader expects exactly one bound child of the root.")
    child = children[0]
    # Per-leaf condition annotations for the root node (must cover every positive-weight leaf — the Loader
    # validates that at construction). The node's weight keys are its leaves over the grouping cols.
    annotations = {root: {tuple(leaf): condition_lookup(tuple(leaf)) for leaf in scheme.nodes[root].weights}}
    loader = Loader(scheme, sampler_config, annotations, preload_to_gpu=preload_to_gpu)
    return _SemanticBatches(loader, root, child, scheme.nodes[root].keys[0])
