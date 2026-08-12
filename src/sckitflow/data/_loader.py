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

from collections.abc import Callable, Collection, Iterator
from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from anndata import AnnData
from scfit.data import EvalLoader as ScfitEvalLoader
from scfit.data import Loader as ScfitLoader
from scfit.data import Stream

from sckitflow.data._utils import with_derived_obs

if TYPE_CHECKING:
    import torch

    # StepData lives in ``core`` (which imports ``data``); import it for typing only to avoid a
    # runtime import cycle. At runtime we build the plain dict directly (see ``_STEP_DATA_KEYS``).
    from sckitflow.core._types import StepData
    from sckitflow.data.schemas import ConditionDataSchema, GroupsDataSchema

# The kwargs contract for these loaders is :class:`~sckitflow.data._manager.LoaderKwargs` -- declared
# there, next to its only consumer (``DataManager.get_dataloaders``), so typing a call site does not
# drag in the scfit/annbatch stack this module imports.
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

# The implicit single group used when the schema declares no groups or conditions at all.
_ALL_CELLS = "_sckitflow_all_cells"


def _state_loc(sample_rep: str | None) -> str:
    """The scfit rep loc for the state representation: ``.X`` (``None``) or an ``obsm`` key."""
    return "X" if sample_rep is None else f"obsm/{sample_rep}"


def _as_tensor(array: Any) -> Any:
    """Return ``array`` as a torch tensor **without copying**, on whatever device it already lives.

    The loaders stream with scfit's ``to=None``, so annbatch hands back its native arrays -- numpy on the
    host, cupy when the read window is GPU-resident. Both map onto torch for free (``from_numpy`` shares
    the buffer; ``__dlpack__`` shares the device allocation), so a GPU-resident window never round-trips
    through host memory. Converting here, rather than via scfit's ``to="torch"``, keeps that one place.
    """
    import torch

    if isinstance(array, torch.Tensor):
        return array
    if isinstance(array, np.ndarray):
        return torch.as_tensor(array)
    return torch.from_dlpack(array)


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

    def __init__(
        self,
        adata: AnnData,
        *,
        condition_schema: ConditionDataSchema,
        groups_schema: GroupsDataSchema,
        sample_rep: str | None = None,
        response_covs: Collection[str] = (),
        pair_key: str | None = None,
        dtype: torch.dtype | None = None,
        device: str | None = None,
        assert_device: bool = True,
    ) -> None:
        # The two schemas that answer a *per-group* question (`_encode_group`, so they outlive construction);
        # everything else the schema declares is read once here, into the locs and columns below.
        self._cond_schema = condition_schema
        self._groups_schema = groups_schema
        self._dtype = dtype
        # Compared by device *type* ("cuda" vs "cuda:0"), matching ``Model.to_device``.
        self._device = device
        self._device_type = device.split(":")[0] if device is not None else None
        self._assert_device = assert_device
        self._state_loc = _state_loc(sample_rep)
        # group_by = group columns + categorical condition columns (continuous covs are streamed reps).
        self._group_cols: tuple[str, ...] = (*groups_schema.groups, *condition_schema.all_condition_cols)
        # Fixed matching (`matched_keys`) rides as one derived column shared by a target group and the source
        # group it flows from -- appended to `group_by` so scfit can `match_on` it, and stripped back off
        # before a leaf is used as a group identity (see `_strip`).
        self._pair_cols: tuple[str, ...] = (pair_key,) if pair_key is not None else ()
        if not self._group_cols:
            # Unconditional schema (no groups, no conditions): scfit must still group on something, so stream
            # one implicit group holding every cell, on a shallow copy of the caller's AnnData.
            adata = with_derived_obs(adata, **{_ALL_CELLS: pd.Categorical(np.full(adata.n_obs, "all"))})
            self._group_cols = (_ALL_CELLS,)
        # Continuous covariates ride as aligned reps: {rep loc -> StepData dict key}.
        self._cond_cont_locs: dict[str, str] = {f"obsm/{c}": c for c in condition_schema.conditions_covariates}
        self._resp_cont_locs: dict[str, str] = {f"obsm/{c}": c for c in response_covs}

        # One O(N) drop_duplicates gives the group combos + a representative row per group (no per-cell logic).
        # The categorical -> encoding is deferred to batch time (cached), keyed by the group's id.
        sub = adata.obs.loc[:, list(self._group_cols)].reset_index(drop=True)
        uniq = sub.drop_duplicates()
        self._leaves: list[tuple] = list(map(tuple, uniq.to_numpy()))
        self._leaf_to_gid: dict[tuple, int] = {leaf: i for i, leaf in enumerate(self._leaves)}
        self._rep_idx: list[int] = uniq.index.to_numpy().tolist()
        self._adata = adata
        self._encode_cache: dict[int, tuple[dict[str, Any], dict[str, Any]]] = {}

    @cached_property
    def _leaf_size(self) -> dict[tuple, int]:
        """Cells per group -- a full-obs groupby, so it is paid on first use rather than at construction.

        Only :class:`EvalLoader` ever asks (to size a metadata-only batch): training reads its row count off
        the delivered tensor. Computing it eagerly meant every training loader -- one per split -- swept the
        whole obs for a number it never read.

        ``groupby(sort=False)`` and the ``drop_duplicates`` behind :attr:`_leaves` both order by first
        appearance, so the counts zip straight onto the leaves (``strict`` makes a divergence loud).
        """
        sub = self._adata.obs.loc[:, list(self._group_cols)]
        sizes = sub.groupby(list(self._group_cols), observed=True, sort=False).size()
        return {leaf: int(n) for leaf, n in zip(self._leaves, sizes.to_numpy(), strict=True)}

    def _conform(self, tensor: Any, *, streamed: bool = True) -> Any:
        """Cast one tensor to the configured ``dtype`` and assert its ``device`` -- the single place both apply.

        ``device`` is checked, never fixed, for a ``streamed`` tensor *when the loader asserts it*: those come
        straight off scfit, which already reads on the streaming device, so a mismatch means the read window is
        on the wrong device and *every* training batch would be copied. That is a real problem and it fails
        here rather than being silently paid for -- the fix is ``preload_to_gpu=True``, scfit's GPU-resident
        read window. :class:`EvalLoader` passes ``assert_device=False``: scfit's eval path reads rows directly,
        so there is no read window to place on the device, its one copy per group is the only option, and
        raising would just make GPU prediction impossible. The per-group ``uns`` encodings are not streamed --
        they are host arrays by construction, so they are moved once, when their cache entry is first filled.
        """
        if self._dtype is not None and tensor.dtype is not self._dtype:
            tensor = tensor.to(self._dtype)
        if self._device_type is not None and tensor.device.type != self._device_type:
            if streamed and self._assert_device:
                raise RuntimeError(
                    f"batches stream on {tensor.device.type!r} but the method runs on {self._device!r}; "
                    "every batch would be copied across devices. Stream on the compute device instead "
                    "(`preload_to_gpu=True`, scfit's GPU-resident read window), or run the method where the "
                    "data is."
                )
            tensor = tensor.to(self._device)
        return tensor

    def _strip(self, leaf: Any, n_prefix: int = 0) -> tuple:
        """The plain group leaf: the streamed leaf without its ``split_by`` prefix or pair-id suffix.

        Both are selectors bolted onto ``group_by`` -- a split excludes cells, a pair id names the source
        group -- and neither is part of the group's identity, so neither may reach ``_leaf_to_gid`` or the
        caller's output ``obs``.
        """
        leaf = tuple(leaf)
        return leaf[n_prefix : len(leaf) - len(self._pair_cols)]

    def _conform_batch(self, step_data: dict[str, Any]) -> dict[str, Any]:
        """One pass over the assembled batch -- the last stage of loading, where dtype/device are settled."""
        if self._dtype is None and self._device is None:
            return step_data
        for key, value in step_data.items():
            if value is None:
                continue
            step_data[key] = (
                {k: self._conform(v) for k, v in value.items()} if isinstance(value, dict) else self._conform(value)
            )
        return step_data

    def _encode_group_cached(self, gid: int) -> tuple[dict[str, Any], dict[str, Any]]:
        """The group's ``(condition, group)`` categorical encodings -- computed once per group, then cached.

        Cached already conformed to ``dtype``/``device``, so the per-batch tile is a device-local expand.
        """
        if gid not in self._encode_cache:
            entry, cond_keys, group_keys = self._encode_group(self._adata[int(self._rep_idx[gid])])
            conformed = {k: self._conform(_as_tensor(v), streamed=False) for k, v in entry.items()}
            self._encode_cache[gid] = (
                {k: conformed[k] for k in cond_keys},
                {k: conformed[k] for k in group_keys},
            )
        return self._encode_cache[gid]

    def _encode_group(self, cell: AnnData) -> tuple[dict[str, np.ndarray], list, list]:
        """Encode one representative group cell to ``{key: vector}`` plus the condition/group key lists."""
        entry: dict[str, np.ndarray] = {}
        cond_keys: list[str] = []
        group_keys: list[str] = []
        condition_data = self._cond_schema.get_data(cell)
        if condition_data is not None and condition_data.categorical_covariates is not None:
            reps = condition_data.categorical_covariates.extract_reps().mapping
            entry.update({k: _leaf_vector(v) for k, v in reps.items()})
            cond_keys = list(reps)
        groups_data = self._groups_schema.get_data(cell)
        if groups_data is not None:
            reps = groups_data.extract_reps().mapping
            entry.update({k: _leaf_vector(v) for k, v in reps.items()})
            group_keys = list(reps)
        return entry, cond_keys, group_keys

    def _batch_size(self, batch: dict, target_state: Any, n_rows: int) -> int:
        """Rows to tile the conditioning to: what the primary delivered, else the caller's known ``n_rows``.

        Measuring the delivered tensor is authoritative -- it already accounts for the sampler's
        ``drop_last`` and for a group shorter than its cap. Only a metadata-only pass (``reps=()``) reads
        no cells at all, and there the caller knows the count outright.
        """
        if target_state is not None:
            return int(target_state.shape[0])
        for loc in (*self._cond_cont_locs, *self._resp_cont_locs):
            if loc in batch[_PRIMARY]:
                return int(batch[_PRIMARY][loc].shape[0])
        return n_rows

    def _align_source(self, source_state: Any, batch_size: int) -> Any:
        """Row-align the source to ``batch_size``, slicing it when longer and tiling it when shorter."""
        if source_state is None or int(source_state.shape[0]) == batch_size:
            return source_state
        n_source = int(source_state.shape[0])
        if n_source == 0:
            raise ValueError(
                "a group was matched to zero control cells, so there is no source to flow from. Check that "
                "every group value present in the primary also has controls (matching is on the group "
                "columns, or on the pair id under `matched_keys`), or predict unpaired by passing "
                "`control_values_dict={}`."
            )
        import torch

        return source_state[torch.arange(batch_size, device=source_state.device) % n_source]

    def _step_data_from_batch(self, batch: dict, gid: int, *, has_state: bool, n_rows: int) -> StepData:
        """Assemble one scfit batch (+ its group id) into a ``StepData``.

        The categorical condition/group encoding is cached per group and tiled to the batch; continuous
        covariates are the per-cell aligned reps already in the batch, and the source is row-aligned to
        the same count. ``has_state=False`` omits the target state (metadata-only prediction), and
        ``n_rows`` is the caller's row count for that case -- see :meth:`_batch_size`.
        """
        target_state = _as_tensor(batch[_PRIMARY][self._state_loc]) if has_state else None
        source_state = _as_tensor(batch[_CONTROL][self._state_loc]) if _CONTROL in batch else None
        batch_size = self._batch_size(batch, target_state, n_rows)
        source_state = self._align_source(source_state, batch_size)

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

        # TODO: the ``*_coupling_lin`` / ``*_coupling_quad`` keys stay None, so `_match_observations` is a
        # no-op and any `match_fn` (OT coupling) never runs. Until then `GenerativeFlow.__init__` refuses a
        # `match_fn` outright, so nobody gets a silently un-coupled run. To lift that: populate these keys --
        # the coupling reps default to the state rep, so it is the same streamed tensor unless `source_rep` /
        # `n_shared_dims` is set -- and verify against OTFM, whose whole point is the coupling.
        step_data: dict[str, Any] = dict.fromkeys(_STEP_DATA_KEYS)
        step_data["target_state"] = target_state
        step_data["source_state"] = source_state
        step_data["target_condition_data"] = condition or None
        step_data["target_group_data"] = group or None
        step_data["target_response_data"] = response or None
        return self._conform_batch(step_data)  # type: ignore[return-value]

    def _control_link(
        self,
        control_adata: AnnData | None,
        control_weights: dict[tuple, float] | None,
        **stream_kwargs: Any,
    ) -> tuple[AnnData | None, Stream | None]:
        # Under fixed matching the pair column *is* the match: one id shared by a target group and the source
        # group it flows from, which is the only thing scfit's value-equality matching can key on. Otherwise a
        # target is matched to whatever shares its group columns.
        match_cols = self._pair_cols or tuple(self._groups_schema.groups)
        # The control stream carries the pair column so `match_on` can reach it; without one it groups on the
        # full group columns (what its weights' keys are).
        group_by = [*self._group_cols, *self._pair_cols]
        if control_adata is not None:
            return control_adata, Stream(
                _CONTROL,
                # A separate pool groups on just what it is matched by, plus the pair column when fixed.
                group_by=group_by if self._pair_cols else list(match_cols or self._group_cols),
                reps=(self._state_loc,),
                match_on=list(match_cols),
                **stream_kwargs,
            )
        if control_weights:
            return None, Stream(
                _PRIMARY,  # same source as the primary; weights pick out the controls
                group_by=group_by,
                reps=(self._state_loc,),
                weights=control_weights,
                match_on=list(match_cols),
                **stream_kwargs,
            )
        return None, None

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

    :param condition_schema: The condition schema: its columns join ``group_by``, its continuous covariates
        stream as reps, and it encodes each group's categorical conditions.
    :type condition_schema: class: `ConditionDataSchema`

    :param groups_schema: The groups schema: its columns lead ``group_by`` and are what a control is matched
        on, and it encodes each group's categorical groups.
    :type groups_schema: class: `GroupsDataSchema`

    :param sample_rep: The ``obsm`` key holding the state representation, or ``None`` for ``.X``.
    :type sample_rep: class: `str | None`

    :param response_covs: Continuous response covariate ``obsm`` keys, streamed per-cell into
        ``target_response_data``.
    :type response_covs: class: `Collection[str]`

    :param split_by: An ``.obs`` column prepended to the primary's ``group_by``, so a leaf is
        ``(split, *group_cols)`` and ``primary_weights`` select cells rather than whole groups. ``None``
        (the default) groups on the group columns alone.
    :type split_by: class: `str | None`

    :param primary_weights: scfit ``{leaf tuple: weight}`` selecting the target (perturbed) groups for
        this loader; ``None`` streams every group uniformly. Controls should be given weight 0 here. The
        tuples are ``(split, *group_cols)`` when ``split_by`` is set, ``group_cols`` otherwise.
    :type primary_weights: class: `dict | None`

    :param control_adata: A separate matched control (source) pool -- faster to load and the
        cross-dataset case. Takes precedence over ``control_weights``.
    :type control_adata: class: `AnnData | None`

    :param control_weights: scfit ``{group tuple: weight}`` selecting controls out of ``adata`` itself,
        used only when ``control_adata`` is ``None``. ``None`` (and no ``control_adata``) => unpaired.
    :type control_weights: class: `dict | None`

    :param pair_key: An ``.obs`` column whose value a target group shares with the source group it flows
        from -- how fixed matching (:class:`DataManager`'s ``matched_keys``) is expressed. Appended to both
        streams' ``group_by`` and matched on, then stripped back off the leaf. ``None`` matches on the group
        columns instead.
    :type pair_key: class: `str | None`

    :param dtype: Torch dtype every emitted tensor is cast to (``None`` = leave as streamed). Set from the
        method's ``dtype`` so a ``float64`` source does not reach ``float32`` modules.
    :type dtype: class: `torch.dtype | None`

    :param device: Device the streamed batches must *already* be on (``None`` = no check). This is an
        assertion, not a move: a mismatch means every batch would be copied, so it raises instead. Pass
        ``preload_to_gpu=True`` to actually stream on a GPU.
    :type device: class: `str | None`

    :param preload_to_gpu: Hand scfit/annbatch a GPU-resident read window, so batches arrive on the device
        instead of being copied there (requires ``sckitflow[gpu]``). Defaults to ``False``.
    :type preload_to_gpu: class: `bool`

    :param n_iters: Number of batches one pass yields. ``None`` = one epoch over the primary. The trainer
        sets this to the training-step count (via :meth:`set_n_iters`) and iterates the loader.
    :type n_iters: class: `int | None`
    """

    def __init__(
        self,
        adata: AnnData,
        *,
        condition_schema: ConditionDataSchema,
        groups_schema: GroupsDataSchema,
        sample_rep: str | None = None,
        response_covs: Collection[str] = (),
        split_by: str | None = None,
        primary_weights: dict[tuple, float] | None = None,
        control_adata: AnnData | None = None,
        control_weights: dict[tuple, float] | None = None,
        pair_key: str | None = None,
        to: str | None = None,
        dtype: torch.dtype | None = None,
        device: str | None = None,
        seed: int = 0,
        n_iters: int | None = None,
        batch_size: int = 128,
        chunk_size: int = 1,
        preload_nchunks: int | None = None,
        preload_to_gpu: bool = False,
    ) -> None:
        super().__init__(
            adata,
            condition_schema=condition_schema,
            groups_schema=groups_schema,
            sample_rep=sample_rep,
            response_covs=response_covs,
            pair_key=pair_key,
            dtype=dtype,
            device=device,
        )
        self._rows_per_batch = batch_size
        sampler_kwargs = {
            "batch_size": batch_size,
            "chunk_size": chunk_size,
            "preload_nchunks": preload_nchunks if preload_nchunks is not None else batch_size // chunk_size,
        }
        primary_reps = (self._state_loc, *self._cond_cont_locs, *self._resp_cont_locs)
        # With a split column the leaf is ``(split, *group_cols)``, so a weight of 0 excludes the *cells*
        # of another split rather than the whole group -- see `DataManager.get_dataloaders`. The split is
        # only a selector, so `_to_step_data` strips it back off before looking the group's encoding up.
        prefix_cols = (split_by,) if split_by is not None else ()
        self._n_prefix = len(prefix_cols)

        primary = Stream(
            _PRIMARY,
            group_by=[*prefix_cols, *self._group_cols, *self._pair_cols],
            reps=primary_reps,
            weights=primary_weights,
            **sampler_kwargs,
        )
        # `self._adata`, not the argument: an unconditional schema streams a shallow copy carrying the
        # implicit all-cells group column (see `with_derived_obs`).
        sources: dict[str, AnnData] = {_PRIMARY: self._adata}
        links: dict[str, Stream] = {}

        # Control link -- `in_memory` materializes just the selected control cells, a small pool re-drawn
        # every batch. The control stream never groups on the split: controls are shared across splits.
        control_source, control_stream = self._control_link(
            control_adata, control_weights, in_memory=True, **sampler_kwargs
        )
        if control_stream is not None:
            links[_CONTROL] = control_stream
            if control_source is not None:
                sources[_CONTROL] = control_source

        # Make the scfit loader finite (one epoch by default) so a plain pass terminates and is
        # re-iterable; the caller (e.g. the trainer) sets the training length via `set_n_iters`.
        self._loader = ScfitLoader(
            sources, primary=primary, links=links, seed=seed, to=to, preload_to_gpu=preload_to_gpu
        )
        self._loader.set_n_iters(n_iters if n_iters is not None else self._loader.n_batches)

    def set_n_iters(self, n_iters: int) -> Loader:
        """Set how many batches one pass yields (delegates to scfit ``set_n_iters``); returns ``self``."""
        if n_iters is None:
            raise ValueError("n_iters must be a positive integer; this loader is always a finite pass.")
        self._loader.set_n_iters(n_iters)
        return self

    def _to_step_data(self, batch: dict) -> StepData:
        """Assemble one sampled scfit batch into a ``StepData``.

        ``batch["leaves"][primary]`` is the group this batch was drawn from; past any ``split_by`` prefix and
        pair-id suffix it is a plain group leaf, which keys the cached categorical encoding.
        """
        leaf = self._strip(batch["leaves"][_PRIMARY], self._n_prefix)
        # every training batch is exactly `batch_size` rows (the sampler drops the short tail)
        return self._step_data_from_batch(batch, self._leaf_to_gid[leaf], has_state=True, n_rows=self._rows_per_batch)

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

    :param condition_schema: The condition schema; see :class:`Loader`.
    :type condition_schema: class: `ConditionDataSchema`

    :param groups_schema: The groups schema; see :class:`Loader`.
    :type groups_schema: class: `GroupsDataSchema`

    :param sample_rep: The ``obsm`` key holding the state representation, or ``None`` for ``.X``.
    :type sample_rep: class: `str | None`

    :param response_covs: Continuous response covariate ``obsm`` keys; see :class:`Loader`.
    :type response_covs: class: `Collection[str]`

    :param require_target_state: When ``False``, the target state is not read (metadata-only prediction);
        ``StepData["target_state"]`` is ``None`` and batches carry only conditioning.
    :type require_target_state: class: `bool`

    :param max_per_group: Per-group cap (``None`` = every cell, ``N`` = at most N, ``1`` = one representative
        per group / dedup). See :class:`scfit.data.EvalLoader`.
    :type max_per_group: class: `int | None`

    :param pair_key: An ``.obs`` column tying a target group to its source group -- fixed matching; see
        :class:`Loader`.
    :type pair_key: class: `str | None`

    :param dtype: Torch dtype every emitted tensor is cast to (``None`` = leave as streamed).
    :type dtype: class: `torch.dtype | None`

    :param device: Device every emitted tensor is moved to (``None`` = leave as streamed). Unlike
        :class:`Loader` this *moves* rather than asserts: scfit's eval path reads rows directly, so there is no
        read window to place on the device and the one copy per group is unavoidable.
    :type device: class: `str | None`
    """

    def __init__(
        self,
        adata: AnnData,
        *,
        condition_schema: ConditionDataSchema,
        groups_schema: GroupsDataSchema,
        sample_rep: str | None = None,
        response_covs: Collection[str] = (),
        primary_weights: dict[tuple, float] | None = None,
        control_adata: AnnData | None = None,
        control_weights: dict[tuple, float] | None = None,
        pair_key: str | None = None,
        require_target_state: bool = True,
        max_per_group: int | None = None,
        subsample: str | Callable = "head",
        to: str | None = None,
        dtype: torch.dtype | None = None,
        device: str | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__(
            adata,
            condition_schema=condition_schema,
            groups_schema=groups_schema,
            sample_rep=sample_rep,
            response_covs=response_covs,
            pair_key=pair_key,
            dtype=dtype,
            device=device,
            assert_device=False,
        )
        self._has_state = require_target_state
        self._max_per_group = max_per_group
        state_reps = (self._state_loc,) if require_target_state else ()
        primary_reps = (*state_reps, *self._cond_cont_locs, *self._resp_cont_locs)

        primary = Stream(
            _PRIMARY,
            group_by=[*self._group_cols, *self._pair_cols],
            reps=primary_reps,
            weights=primary_weights,
        )
        sources: dict[str, AnnData] = {_PRIMARY: self._adata}
        links: dict[str, Stream] = {}

        # Control link (source), matched on the group columns -- controls always carry a state to flow from.
        control_source, control_stream = self._control_link(control_adata, control_weights)
        if control_stream is not None:
            links[_CONTROL] = control_stream
            if control_source is not None:
                sources[_CONTROL] = control_source

        self._loader = ScfitEvalLoader(
            sources, primary=primary, links=links, to=to, max_per_group=max_per_group, subsample=subsample, seed=seed
        )

    def __iter__(self) -> Iterator[tuple[StepData, tuple]]:
        """Yield ``(StepData, leaf)`` per selected group, deterministically."""
        for batch in self._loader:
            leaf = self._strip(batch["leaf"])
            # the group's cell count, capped the same way scfit caps the rows it reads
            n_rows = self._leaf_size[leaf]
            if self._max_per_group is not None:
                n_rows = min(n_rows, self._max_per_group)
            yield self._step_data_from_batch(batch, self._leaf_to_gid[leaf], has_state=self._has_state, n_rows=n_rows), leaf

    def __len__(self) -> int:
        """Number of groups (== number of batches) this pass yields."""
        return len(self._loader)
