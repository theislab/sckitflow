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
import pandas as pd
from anndata import AnnData
from scfit.data import EvalLoader as ScfitEvalLoader
from scfit.data import Loader as ScfitLoader
from scfit.data import Stream

if TYPE_CHECKING:
    import torch

    # StepData lives in ``core`` (which imports ``data``); import it for typing only to avoid a
    # runtime import cycle. At runtime we build the plain dict directly (see ``_STEP_DATA_KEYS``).
    from sckitflow.core._types import StepData
    from sckitflow.data._manager import DataManager

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
        dm: DataManager,
        *,
        dtype: torch.dtype | None = None,
        device: str | None = None,
    ) -> None:
        self._dm = dm
        self._dtype = dtype
        # Compared by device *type* ("cuda" vs "cuda:0"), matching ``Model.to_device``.
        self._device = device
        self._device_type = device.split(":")[0] if device is not None else None
        cond_schema = dm.condition_data_schema
        self._state_loc = _state_loc(dm.state_data_schema.sample_rep)
        # group_by = group columns + categorical condition columns (continuous covs are streamed reps).
        self._group_cols: tuple[str, ...] = dm.group_cols
        if not self._group_cols:
            # Unconditional schema (no groups, no conditions): scfit must still group on something, so
            # stream one implicit group holding every cell. Written onto `adata.obs` because that is
            # where scfit reads grouping from; it is constant and idempotent.
            if _ALL_CELLS not in adata.obs:
                adata.obs[_ALL_CELLS] = pd.Categorical(np.full(adata.n_obs, "all"))
            self._group_cols = (_ALL_CELLS,)
        # Continuous covariates ride as aligned reps: {rep loc -> StepData dict key}.
        self._cond_cont_locs: dict[str, str] = {f"obsm/{c}": c for c in cond_schema.conditions_covariates}
        self._resp_cont_locs: dict[str, str] = {f"obsm/{c}": c for c in dm.target_data_schema.continuous_covs}

        # One O(N) drop_duplicates gives the group combos + a representative row per group (no per-cell logic).
        # The categorical -> encoding is deferred to batch time (cached), keyed by the group's id.
        sub = adata.obs.loc[:, list(self._group_cols)].reset_index(drop=True)
        uniq = sub.drop_duplicates()
        self._leaves: list[tuple] = list(map(tuple, uniq.to_numpy()))
        self._leaf_to_gid: dict[tuple, int] = {leaf: i for i, leaf in enumerate(self._leaves)}
        self._rep_idx: list[int] = uniq.index.to_numpy().tolist()
        # Cells per group. `groupby(sort=False)` and `drop_duplicates` both order by first appearance, so
        # the counts zip straight onto the leaves (`strict` makes a divergence loud rather than silent).
        sizes = sub.groupby(list(self._group_cols), observed=True, sort=False).size()
        self._leaf_size: dict[tuple, int] = {
            leaf: int(n) for leaf, n in zip(self._leaves, sizes.to_numpy(), strict=True)
        }
        self._adata = adata
        self._encode_cache: dict[int, tuple[dict[str, Any], dict[str, Any]]] = {}

    def _conform(self, tensor: Any, *, streamed: bool = True) -> Any:
        """Cast one tensor to the configured ``dtype`` and assert its ``device`` -- the single place both apply.

        ``device`` is checked, never fixed, for a ``streamed`` tensor: those come straight off scfit, which
        already reads on the streaming device, so a mismatch means the read window is on the wrong device
        and *every* batch would be copied. That is a real problem and it fails here rather than being
        silently paid for. The per-group ``uns`` encodings are not streamed -- they are host arrays by
        construction, so they are moved once, when their cache entry is first filled.
        """
        if self._dtype is not None and tensor.dtype is not self._dtype:
            tensor = tensor.to(self._dtype)
        if self._device_type is not None and tensor.device.type != self._device_type:
            if streamed:
                raise RuntimeError(
                    f"batches stream on {tensor.device.type!r} but the method runs on {self._device!r}; "
                    "every batch would be copied across devices. Stream on the compute device instead "
                    "(scfit's GPU-resident read window), or run the method where the data is."
                )
            tensor = tensor.to(self._device)
        return tensor

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
                "columns), or predict unpaired by passing `control_values_dict=None`."
            )
        import torch

        return source_state[torch.arange(batch_size, device=source_state.device) % n_source]

    def _assemble(self, batch: dict, gid: int, *, has_state: bool, n_rows: int) -> StepData:
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
        # no-op and any `match_fn` (OT coupling) never runs. Populate them -- the coupling reps default to
        # the state rep, so it is the same streamed tensor unless `source_rep` / `n_shared_dims` is set --
        # and verify against OTFM, whose whole point is the coupling.
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
        match_cols = tuple(self._dm.groups_data_schema.groups)
        if control_adata is not None:
            return control_adata, Stream(
                _CONTROL,
                group_by=list(match_cols) if match_cols else list(self._group_cols),
                reps=(self._state_loc,),
                match_on=list(match_cols),
                **stream_kwargs,
            )
        if control_weights:
            return None, Stream(
                _PRIMARY,  # same source as the primary; weights pick out the controls
                group_by=list(self._group_cols),
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

    :param dm: The data manager whose schema defines the state/condition/group layout.
    :type dm: class: `DataManager`

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

    :param dtype: Torch dtype every emitted tensor is cast to (``None`` = leave as streamed). Set from the
        method's ``dtype`` so a ``float64`` source does not reach ``float32`` modules.
    :type dtype: class: `torch.dtype | None`

    :param device: Device the streamed batches must *already* be on (``None`` = no check). This is an
        assertion, not a move: a mismatch means every batch would be copied, so it raises instead.
    :type device: class: `str | None`

    :param n_iters: Number of batches one pass yields. ``None`` = one epoch over the primary. The trainer
        sets this to the training-step count (via :meth:`set_n_iters`) and iterates the loader.
    :type n_iters: class: `int | None`
    """

    def __init__(
        self,
        adata: AnnData,
        *,
        dm: DataManager,
        split_by: str | None = None,
        primary_weights: dict[tuple, float] | None = None,
        control_adata: AnnData | None = None,
        control_weights: dict[tuple, float] | None = None,
        to: str | None = None,
        dtype: torch.dtype | None = None,
        device: str | None = None,
        seed: int = 0,
        n_iters: int | None = None,
        batch_size: int = 128,
        chunk_size: int = 1,
        preload_nchunks: int | None = None,
    ) -> None:
        super().__init__(adata, dm, dtype=dtype, device=device)
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
            group_by=[*prefix_cols, *self._group_cols],
            reps=primary_reps,
            weights=primary_weights,
            **sampler_kwargs,
        )
        sources: dict[str, AnnData] = {_PRIMARY: adata}
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
        self._loader = ScfitLoader(sources, primary=primary, links=links, seed=seed, to=to)
        self._loader.set_n_iters(n_iters if n_iters is not None else self._loader.n_batches)

    def set_n_iters(self, n_iters: int) -> Loader:
        """Set how many batches one pass yields (delegates to scfit ``set_n_iters``); returns ``self``."""
        if n_iters is None:
            raise ValueError("n_iters must be a positive integer; this loader is always a finite pass.")
        self._loader.set_n_iters(n_iters)
        return self

    def _to_step_data(self, batch: dict) -> StepData:
        """Assemble one sampled scfit batch into a ``StepData``.

        ``batch["leaves"][primary]`` is the group this batch was drawn from; past any ``split_by`` prefix
        it is a plain group leaf, which keys the cached categorical encoding.
        """
        leaf = tuple(batch["leaves"][_PRIMARY])[self._n_prefix :]
        # every training batch is exactly `batch_size` rows (the sampler drops the short tail)
        return self._assemble(batch, self._leaf_to_gid[leaf], has_state=True, n_rows=self._rows_per_batch)

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

    :param dtype: Torch dtype every emitted tensor is cast to (``None`` = leave as streamed).
    :type dtype: class: `torch.dtype | None`

    :param device: Device the streamed batches must already be on (``None`` = no check); see :class:`Loader`.
    :type device: class: `str | None`
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
        dtype: torch.dtype | None = None,
        device: str | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__(adata, dm, dtype=dtype, device=device)
        self._has_state = require_target_state
        self._max_per_group = max_per_group
        state_reps = (self._state_loc,) if require_target_state else ()
        primary_reps = (*state_reps, *self._cond_cont_locs, *self._resp_cont_locs)

        primary = Stream(_PRIMARY, group_by=list(self._group_cols), reps=primary_reps, weights=primary_weights)
        sources: dict[str, AnnData] = {_PRIMARY: adata}
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
            leaf = tuple(batch["leaf"])
            # the group's cell count, capped the same way scfit caps the rows it reads
            n_rows = self._leaf_size[leaf]
            if self._max_per_group is not None:
                n_rows = min(n_rows, self._max_per_group)
            yield self._assemble(batch, self._leaf_to_gid[leaf], has_state=self._has_state, n_rows=n_rows), leaf

    def __len__(self) -> int:
        """Number of groups (== number of batches) this pass yields."""
        return len(self._loader)
