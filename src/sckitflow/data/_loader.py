r"""scfit-batch to ``StepData`` adapters: :class:`Loader` (training) and :class:`EvalLoader` (prediction).

Selection is entirely scfit-native: one AnnData is streamed and the split + perturbed/control choice is
expressed as scfit **weights** over scfit's own leaf factorization (weight 0 == excluded, ``in_memory``
materializes only the selected cells). No Python-side masking or copying of subsets.

* the **primary** stream is the target/perturbed population -- controls get weight 0;
* the **control** link is the source population -- a separate ``control_adata`` (fast to load, and the
  cross-dataset case), or, when none is given, the *same* adata weighted so only controls are nonzero;
* **continuous** covariates ride as *aligned reps* (per-cell, straight from ``obsm``);
* **categorical** condition/group encodings are computed per group at batch time (cached), keyed off the
  group the batch belongs to, then tiled to the batch.

:class:`Loader` samples a stochastic per-batch schedule for training (one group per ``__next__``).
:class:`EvalLoader` walks every group once, deterministically, for prediction -- yielding ``(StepData,
leaf)`` where ``leaf`` is the group's ``group_by`` value tuple, enough to rebuild the output ``obs``
without any ``ann_df`` / container round-trip.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

import numpy as np
from anndata import AnnData
from scfit.data import EvalLoader as ScfitEvalLoader
from scfit.data import Loader as ScfitLoader
from scfit.data import Stream

if TYPE_CHECKING:
    # StepData lives in ``core`` (which imports ``data``); import it for typing only to avoid a
    # runtime import cycle. At runtime we build the plain dict directly (see ``_STEP_DATA_KEYS``).
    from sckitflow.core._types import StepData
    from sckitflow.data._manager import DataManager

__all__ = ["Loader", "EvalLoader"]

# Every StepData key, so the emitted dict is complete and consumers can index without guarding.
_STEP_DATA_KEYS = (
    "target_state",
    "target_coupling_lin",
    "target_coupling_quad",
    "target_condition_data",
    "target_group_data",
    "target_response_data",
    "source_state",
    "source_coupling_lin",
    "source_coupling_quad",
    "source_condition_data",
    "source_group_data",
)

_PRIMARY = "primary"
_CONTROL = "control"


def _state_loc(sample_rep: str | None) -> str:
    """The scfit rep loc for the state representation: ``.X`` (``None``) or an ``obsm`` key."""
    return "X" if sample_rep is None else f"obsm/{sample_rep}"


def _as_tensor(array: Any) -> Any:
    """Return ``array`` as a torch tensor (a no-op when scfit already emitted torch via ``to``)."""
    import torch

    return array if isinstance(array, torch.Tensor) else torch.as_tensor(np.asarray(array))


def _tile(vector: Any, batch_size: int) -> Any:
    """Broadcast a per-group encoding ``(dim,)`` to a per-cell batch ``(batch_size, dim)``."""
    tensor = _as_tensor(vector)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor.expand(batch_size, *tensor.shape[1:]).contiguous()


def _leaf_vector(rep: Any) -> np.ndarray:
    """One representative encoding ``(dim,)`` for a group leaf (rows are identical within a group)."""
    array = np.asarray(rep)
    return array[0] if array.ndim >= 2 else array


class _StepDataBridge:
    """Shared scfit-batch -> ``StepData`` machinery for :class:`Loader` and :class:`EvalLoader`.

    Owns the schema-derived rep locations, the ordered group leaves + a representative row per group, the
    per-group categorical encoding (cached), and the assembly of one scfit batch into a ``StepData``.
    Subclasses build the concrete scfit loader (sampling vs deterministic) and drive iteration.
    """

    def __init__(self, adata: AnnData, dm: DataManager) -> None:
        self._dm = dm
        cond_schema = dm.condition_data_schema
        self._state_loc = _state_loc(dm.state_data_schema.sample_rep)
        # group_by = group columns + categorical condition columns (continuous covs are streamed reps).
        self._group_cols: tuple[str, ...] = (*tuple(dm.groups_data_schema.groups), *cond_schema.all_condition_cols)
        if not self._group_cols:
            raise ValueError(
                "Loader needs at least one categorical group/condition column to group on "
                "(set `groups=` and/or `conditions=` on the DataManager)."
            )
        # Continuous covariates ride as aligned reps: {rep loc -> StepData dict key}.
        self._cond_cont_locs: dict[str, str] = {f"obsm/{c}": c for c in cond_schema.conditions_covariates}
        self._resp_cont_locs: dict[str, str] = {f"obsm/{c}": c for c in dm.target_data_schema.continuous_covs}

        # One O(N) drop_duplicates gives the group combos + a representative row per group (no per-cell logic).
        # The categorical -> encoding is deferred to batch time (cached), keyed by the group's id.
        uniq = adata.obs.loc[:, list(self._group_cols)].reset_index(drop=True).drop_duplicates()
        self._leaves: list[tuple] = list(map(tuple, uniq.to_numpy()))
        self._leaf_to_gid: dict[tuple, int] = {leaf: i for i, leaf in enumerate(self._leaves)}
        self._rep_idx: list[int] = uniq.index.to_numpy().tolist()
        self._adata = adata
        self._encode_cache: dict[int, tuple[dict[str, np.ndarray], dict[str, np.ndarray]]] = {}

    def _encode_group_cached(self, gid: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """The group's ``(condition, group)`` categorical encodings -- computed once per group, then cached."""
        if gid not in self._encode_cache:
            entry, cond_keys, group_keys = self._encode_group(self._adata[int(self._rep_idx[gid])])
            self._encode_cache[gid] = ({k: entry[k] for k in cond_keys}, {k: entry[k] for k in group_keys})
        return self._encode_cache[gid]

    def _encode_group(self, cell: AnnData) -> tuple[dict[str, np.ndarray], list, list]:
        """Encode one representative group cell to ``{key: vector}`` plus the condition/group key lists."""
        entry: dict[str, np.ndarray] = {}
        cond_keys: list[str] = []
        group_keys: list[str] = []
        condition_data = self._dm.condition_data_schema.get_data(cell)
        if condition_data is not None and condition_data.categorical_covariates is not None:
            reps = condition_data.categorical_covariates.extract_reps().mapping
            entry.update({k: _leaf_vector(v) for k, v in reps.items()})
            cond_keys = list(reps)
        groups_data = self._dm.groups_data_schema.get_data(cell)
        if groups_data is not None:
            reps = groups_data.extract_reps().mapping
            entry.update({k: _leaf_vector(v) for k, v in reps.items()})
            group_keys = list(reps)
        return entry, cond_keys, group_keys

    def _batch_size(self, batch: dict, target_state: Any) -> int:
        """Rows in the batch: from the target state, else a per-cell continuous rep, else 1 (metadata-only)."""
        if target_state is not None:
            return int(target_state.shape[0])
        for loc in (*self._cond_cont_locs, *self._resp_cont_locs):
            if loc in batch[_PRIMARY]:
                return int(np.asarray(batch[_PRIMARY][loc]).shape[0])
        return 1

    def _assemble(self, batch: dict, gid: int, *, has_state: bool) -> StepData:
        """Assemble one scfit batch (+ its group id) into a ``StepData``.

        The categorical condition/group encoding is cached per group and tiled to the batch; continuous
        covariates are the per-cell aligned reps already in the batch. ``has_state=False`` omits the target
        state (metadata-only prediction).
        """
        target_state = _as_tensor(batch[_PRIMARY][self._state_loc]) if has_state else None
        source_state = _as_tensor(batch[_CONTROL][self._state_loc]) if _CONTROL in batch else None
        batch_size = self._batch_size(batch, target_state)

        cond_enc, group_enc = self._encode_group_cached(gid)
        condition: dict[str, Any] = {k: _tile(v, batch_size) for k, v in cond_enc.items()}
        group: dict[str, Any] = {k: _tile(v, batch_size) for k, v in group_enc.items()}
        # continuous covariates: per-cell aligned reps (present only when the stream carries them)
        for loc, key in self._cond_cont_locs.items():
            if loc in batch[_PRIMARY]:
                condition[key] = _as_tensor(batch[_PRIMARY][loc])
        response = {
            key: _as_tensor(batch[_PRIMARY][loc]) for loc, key in self._resp_cont_locs.items() if loc in batch[_PRIMARY]
        }

        step_data: dict[str, Any] = dict.fromkeys(_STEP_DATA_KEYS)
        step_data["target_state"] = target_state
        step_data["source_state"] = source_state
        step_data["target_condition_data"] = condition or None
        step_data["target_group_data"] = group or None
        step_data["target_response_data"] = response or None
        return step_data  # type: ignore[return-value]

    @property
    def group_cols(self) -> tuple[str, ...]:
        """The ``group_by`` columns (groups then categorical conditions) -- the order of a ``leaf`` tuple."""
        return self._group_cols

    @property
    def cond_cont_keys(self) -> tuple[str, ...]:
        """Continuous condition covariate keys carried per-cell in ``target_condition_data``."""
        return tuple(self._cond_cont_locs.values())

    @property
    def resp_keys(self) -> tuple[str, ...]:
        """Continuous response covariate keys carried per-cell in ``target_response_data``."""
        return tuple(self._resp_cont_locs.values())


class Loader(_StepDataBridge):
    """Yields ``StepData`` batches for one population, backed by :class:`scfit.data.Loader`.

    :param adata: The annotated data object to stream (the whole thing -- selection is by weights).
    :type adata: class: `AnnData`

    :param dm: The data manager whose schema defines the state/condition/group layout.
    :type dm: class: `DataManager`

    :param primary_weights: scfit ``{group tuple: weight}`` selecting the target (perturbed) groups for
        this loader; ``None`` streams every group uniformly. Controls should be given weight 0 here.
    :type primary_weights: class: `dict | None`

    :param control_adata: A separate matched control (source) pool -- faster to load and the
        cross-dataset case. Takes precedence over ``control_weights``.
    :type control_adata: class: `AnnData | None`

    :param control_weights: scfit ``{group tuple: weight}`` selecting controls out of ``adata`` itself,
        used only when ``control_adata`` is ``None``. ``None`` (and no ``control_adata``) => unpaired.
    :type control_weights: class: `dict | None`

    :param n_iters: Number of batches one pass yields. ``None`` = one epoch over the primary. The trainer
        sets this to the training-step count (via :meth:`set_n_iters`) and iterates the loader.
    :type n_iters: class: `int | None`
    """

    def __init__(
        self,
        adata: AnnData,
        *,
        dm: DataManager,
        primary_weights: dict[tuple, float] | None = None,
        control_adata: AnnData | None = None,
        control_weights: dict[tuple, float] | None = None,
        to: str | None = None,
        seed: int = 0,
        n_iters: int | None = None,
        batch_size: int = 128,
        chunk_size: int = 1,
        preload_nchunks: int | None = None,
    ) -> None:
        super().__init__(adata, dm)
        sampler_kwargs = {
            "batch_size": batch_size,
            "chunk_size": chunk_size,
            "preload_nchunks": preload_nchunks if preload_nchunks is not None else batch_size // chunk_size,
        }
        primary_reps = (self._state_loc, *self._cond_cont_locs, *self._resp_cont_locs)

        # scfit surfaces the drawn group per batch via label_lookup; carry a group id so the (cached)
        # categorical encoding can be looked up at batch time (see `_to_step_data`).
        wanted = set(primary_weights) if primary_weights else set(self._leaves)
        label_lookup = {
            leaf: {"gid": np.array([gid], dtype=np.int64)} for leaf, gid in self._leaf_to_gid.items() if leaf in wanted
        }

        primary = Stream(
            _PRIMARY,
            group_by=list(self._group_cols),
            reps=primary_reps,
            weights=primary_weights,
            label_lookup=label_lookup or None,
            **sampler_kwargs,
        )
        sources: dict[str, AnnData] = {_PRIMARY: adata}
        links: dict[str, Stream] = {}

        # Control link, matched on the group columns (e.g. cell line). Either a separate pool, or the same
        # adata weighted to controls-only -- scfit's in_memory materializes just those selected cells. When
        # neither is given the setting is unpaired: no control link, so ``source_state`` stays ``None``.
        match_cols = tuple(dm.groups_data_schema.groups)
        if control_adata is not None:
            sources[_CONTROL] = control_adata
            links[_CONTROL] = Stream(
                _CONTROL,
                group_by=list(match_cols) if match_cols else list(self._group_cols),
                reps=(self._state_loc,),
                match_on=list(match_cols),
                in_memory=True,
                **sampler_kwargs,
            )
        elif control_weights:
            links[_CONTROL] = Stream(
                _PRIMARY,  # same source as the primary; weights pick out the controls
                group_by=list(self._group_cols),
                reps=(self._state_loc,),
                weights=control_weights,
                match_on=list(match_cols),
                in_memory=True,
                **sampler_kwargs,
            )
        # Make the scfit loader finite (one epoch by default) so a plain pass terminates and is
        # re-iterable; the caller (e.g. the trainer) sets the training length via `set_n_iters`.
        self._loader = ScfitLoader(sources, primary=primary, links=links, seed=seed, to=to)
        self._loader.set_n_iters(n_iters if n_iters is not None else self._loader.n_batches)

    def set_n_iters(self, n_iters: int) -> Loader:
        """Set how many batches one pass yields (delegates to scfit ``set_n_iters``); returns ``self``."""
        self._loader.set_n_iters(n_iters)
        return self

    def _to_step_data(self, batch: dict) -> StepData:
        """Assemble one sampled scfit batch into a ``StepData`` (group id read from ``batch["labels"]``)."""
        gid = int(batch["labels"][_PRIMARY]["gid"][0])
        return self._assemble(batch, gid, has_state=True)

    def __iter__(self) -> Iterator[StepData]:
        """One finite pass of ``StepData`` batches; re-iterable, so the trainer just iterates it."""
        for batch in self._loader:
            yield self._to_step_data(batch)

    def __len__(self) -> int:
        """Number of batches one pass yields (scfit's ``n_iters``, set finite in ``__init__``)."""
        return int(self._loader.n_iters)


class EvalLoader(_StepDataBridge):
    """Deterministic, per-group ``(StepData, leaf)`` for prediction, backed by :class:`scfit.data.EvalLoader`.

    Walks every selected primary group once (full coverage, or a ``max_per_group`` cap that at ``1`` becomes
    predict-once-per-condition), each matched to its control leaf. ``leaf`` is the group's ``group_by`` value
    tuple -- ordered as :attr:`group_cols` -- from which the caller rebuilds the output ``obs`` directly.

    :param require_target_state: When ``False``, the target state is not read (metadata-only prediction);
        ``StepData["target_state"]`` is ``None`` and batches carry only conditioning.
    :type require_target_state: class: `bool`

    :param max_per_group: Per-group cap (``None`` = every cell, ``N`` = at most N, ``1`` = one representative
        per group / dedup). See :class:`scfit.data.EvalLoader`.
    :type max_per_group: class: `int | None`
    """

    def __init__(
        self,
        adata: AnnData,
        *,
        dm: DataManager,
        primary_weights: dict[tuple, float] | None = None,
        control_adata: AnnData | None = None,
        control_weights: dict[tuple, float] | None = None,
        require_target_state: bool = True,
        max_per_group: int | None = None,
        subsample: str | Callable = "head",
        to: str | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__(adata, dm)
        self._has_state = require_target_state
        state_reps = (self._state_loc,) if require_target_state else ()
        primary_reps = (*state_reps, *self._cond_cont_locs, *self._resp_cont_locs)

        primary = Stream(_PRIMARY, group_by=list(self._group_cols), reps=primary_reps, weights=primary_weights)
        sources: dict[str, AnnData] = {_PRIMARY: adata}
        links: dict[str, Stream] = {}

        # Control link (source), matched on the group columns -- controls always carry a state to flow from.
        match_cols = tuple(dm.groups_data_schema.groups)
        if control_adata is not None:
            sources[_CONTROL] = control_adata
            links[_CONTROL] = Stream(
                _CONTROL,
                group_by=list(match_cols) if match_cols else list(self._group_cols),
                reps=(self._state_loc,),
                match_on=list(match_cols),
            )
        elif control_weights:
            links[_CONTROL] = Stream(
                _PRIMARY,
                group_by=list(self._group_cols),
                reps=(self._state_loc,),
                weights=control_weights,
                match_on=list(match_cols),
            )
        self._loader = ScfitEvalLoader(
            sources, primary=primary, links=links, to=to, max_per_group=max_per_group, subsample=subsample, seed=seed
        )

    def __iter__(self) -> Iterator[tuple[StepData, tuple]]:
        """Yield ``(StepData, leaf)`` per selected group, deterministically."""
        for batch in self._loader:
            leaf = tuple(batch["leaf"])
            yield self._assemble(batch, self._leaf_to_gid[leaf], has_state=self._has_state), leaf

    def __len__(self) -> int:
        """Number of groups (== number of batches) this pass yields."""
        return len(self._loader)
