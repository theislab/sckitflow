r"""scfit-batch to ``StepData`` adapters: :class:`Loader` (training) and :class:`EvalLoader` (prediction).

Selection is entirely scfit-native: one AnnData is streamed and the split + perturbed/control choice is
expressed as scfit **weights** over scfit's own leaf factorization (weight 0 == excluded, ``in_memory``
materializes only the selected observations). No Python-side masking or copying of subsets.

* the **primary** stream is the target/perturbed population -- controls get weight 0;
* the **control** link is the source population -- the *same* adata weighted so only controls are nonzero,
  held in memory by scfit;
* **continuous** covariates ride as *aligned reps* (per-obs, straight from ``obsm``);
* **categorical** condition/group encodings are computed per group at batch time (cached), keyed off the
  group the batch belongs to, as one row ``(1, c, d)``.

:class:`Loader` samples a stochastic per-batch schedule for training (one group per ``__next__``).
:class:`EvalLoader` walks every group once, deterministically, for prediction -- yielding ``(StepData,
leaf)`` where ``leaf`` is the group's ``group_by`` value tuple, enough to rebuild the output ``obs``
without any ``ann_df`` / container round-trip.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable, Iterator
from functools import cached_property
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd
from anndata import AnnData
from scfit.data import EvalLoader as ScfitEvalLoader
from scfit.data import Loader as ScfitLoader
from scfit.data import SamplerParams, Stream, obs_columns

from sckitflow.data._utils import AnnDataLike, as_adatas


@runtime_checkable
class SupportsDLPack(Protocol):
    """Any array that can hand over its buffer without a copy."""

    def __dlpack__(self, *, stream: int | None = None) -> Any: ...


if TYPE_CHECKING:
    import torch

    # StepData lives in ``core`` (which imports ``data``); import it for typing only to avoid a
    # runtime import cycle. At runtime we build the plain dict directly (see ``_STEP_DATA_KEYS``).
    from sckitflow.core._types import StepData
    from sckitflow.data.schemas import ConditionDataSchema, CouplingDataSchema, GroupsDataSchema

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
    "n_obs",
)

_PRIMARY = "primary"
_CONTROL = "control"

# Coupling terms
_TARGET_LIN_TERM = "target_coupling_lin"
_TARGET_QUAD_TERM = "target_coupling_quad"
_SOURCE_LIN_TERM = "source_coupling_lin"
_SOURCE_QUAD_TERM = "source_coupling_quad"


def _state_loc(sample_rep: str | None) -> str:
    """The scfit rep loc for the state representation: ``.X`` (``None``) or an ``obsm`` key."""
    return "X" if sample_rep is None else f"obsm/{sample_rep}"


def _as_tensor(array: SupportsDLPack) -> torch.Tensor:
    """Any streamed array -> ``torch.Tensor`` sharing its buffer, on the device it already lives."""
    import torch

    if isinstance(array, torch.Tensor):
        return array
    if isinstance(array, np.ndarray):
        return torch.as_tensor(array)
    if not isinstance(array, SupportsDLPack):
        raise TypeError(
            f"cannot convert a streamed {type(array).__name__} to a torch tensor without copying: expected "
            "a numpy/torch array, or an array exposing the DLPack protocol (`__dlpack__`, e.g. cupy on a "
            "GPU-resident read window). Check that the streamed rep holds a dense array."
        )
    return torch.from_dlpack(array)


def _leaf_encoding(array: np.ndarray) -> np.ndarray:
    """One group leaf's encoding ``(1, c, d)`` -> ``(c, d)``; a flat ``(d,)`` passes through.

    ``c`` is the columns mapped to the realm, ``d`` the encoding width (:meth:`extract_reps`). The leading
    axis is the single representative observation the encoding came from, so anything above 1 is a bug,
    not a set.
    """
    if array.ndim == 1:
        return array
    if array.shape[0] != 1:
        raise ValueError(
            f"expected one representative row per group encoding, got {array.shape[0]} (shape {array.shape}). "
            "A group leaf is encoded from a single observation, so the leading dimension must be 1."
        )
    if array.ndim != 3:
        # `extract_reps` always emits (n_obs, c, d); another rank means it did not come from there.
        raise ValueError(f"expected a group encoding of rank 1 or 3, got shape {array.shape}.")
    return array[0]


def _group_row(encoding: torch.Tensor) -> torch.Tensor:
    """One group's encoding ``(c, d)`` -> ``(1, c, d)``; ``c`` is the axis :class:`SetEncoder` pools."""
    if encoding.ndim == 1:  # a flat (d,) rep carries no set axis of its own
        encoding = encoding.unsqueeze(0)
    elif encoding.ndim != 2:
        # Rank 3 here means the per-observation axis was never dropped -- `_leaf_encoding` was skipped -- and
        # the batch axis would land in front of it, handing the encoder a 4-D tensor to pool.
        raise ValueError(f"expected an encoding of rank 1 or 2, got shape {tuple(encoding.shape)}.")
    return encoding.unsqueeze(0)


def _get_coupling_locs(coupling_schema: CouplingDataSchema | None) -> tuple[dict[str, str], dict[str, str]]:
    """Retrieves the locations from the input coupling schema."""
    # Return empty dictionaries
    if coupling_schema is None:
        return {}, {}

    # Initialize dictionaries.
    source_coupling_locs: dict[str, str] = {}
    target_coupling_locs: dict[str, str] = {}

    # Linear term, both source and target
    if coupling_schema.use_linear:
        loc_lin = _state_loc(coupling_schema.rep_lin)
        source_coupling_locs[loc_lin] = _SOURCE_LIN_TERM
        target_coupling_locs[loc_lin] = _TARGET_LIN_TERM

    # Source quadratic term.
    if coupling_schema.source_rep_quad:
        source_coupling_locs[f"obsm/{coupling_schema.source_rep_quad}"] = _SOURCE_QUAD_TERM

    # Target quadratic term.
    if coupling_schema.target_rep_quad:
        target_coupling_locs[f"obsm/{coupling_schema.target_rep_quad}"] = _TARGET_QUAD_TERM
    return source_coupling_locs, target_coupling_locs


# The schema field each coupling term is read from, for error messages.
_COUPLING_FIELD = {
    _SOURCE_LIN_TERM: "rep_lin",
    _TARGET_LIN_TERM: "rep_lin",
    _SOURCE_QUAD_TERM: "source_rep_quad",
    _TARGET_QUAD_TERM: "target_rep_quad",
}


def get_rep(adata: AnnData, loc: str) -> Any:
    """Rep ``loc`` (``"X"`` | ``"obsm/<k>"`` | ``"layers/<k>"``) of ``adata``."""
    if loc == "X":
        return adata.X
    field, key = loc.split("/", 1)
    return getattr(adata, field)[key]


def _has_rep(adata: AnnData, loc: str) -> bool:
    """Whether ``adata`` carries rep ``loc`` (``"X"`` | ``"obsm/<k>"`` | ``"layers/<k>"``)."""
    if loc == "X":
        return adata.X is not None
    field, key = loc.split("/", 1)
    return key in getattr(adata, field)


def _merge_reps(*groups: Iterable[str]) -> tuple[str, ...]:
    seen: dict[str, None] = {}
    for g in groups:
        for loc in g:
            seen.setdefault(loc, None)
    return tuple(seen)


class _StepDataBridge:
    """Shared scfit-batch -> ``StepData`` machinery for :class:`Loader` and :class:`EvalLoader`.

    Owns the schema-derived rep locations, the ordered group leaves + a representative row per group, the
    per-group categorical encoding (cached), and the assembly of one scfit batch into a ``StepData``.
    Subclasses build the concrete scfit loader (sampling vs deterministic) and drive iteration.
    """

    def __init__(
        self,
        adata: AnnDataLike,
        *,
        condition_schema: ConditionDataSchema,
        groups_schema: GroupsDataSchema,
        coupling_schema: CouplingDataSchema | None = None,
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
        self._coupling_schema = coupling_schema
        self._dtype = dtype
        # Compared by device *type* ("cuda" vs "cuda:0"), matching the module's own device.
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
        adatas = as_adatas(adata)
        # Continuous covariates ride as aligned reps: {rep loc -> StepData dict key}.
        self._cond_cont_locs: dict[str, str] = {f"obsm/{c}": c for c in condition_schema.continuous_conditions}
        self._resp_cont_locs: dict[str, str] = {f"obsm/{c}": c for c in response_covs}

        # Handle coupling terms and construct coupling locs dictionary
        self._coupling_schema = coupling_schema
        self._source_coupling_locs, self._target_coupling_locs = _get_coupling_locs(self._coupling_schema)
        # Which schema field asks for each rep, so a missing one is reported by the name the user set.
        self._rep_fields: dict[str, list[str]] = {}
        for loc, field in [
            (self._state_loc, "sample_rep"),
            *((loc, "continuous_conditions") for loc in self._cond_cont_locs),
            *((loc, "response covariates") for loc in self._resp_cont_locs),
            *((loc, _COUPLING_FIELD[t]) for loc, t in self._source_coupling_locs.items()),
            *((loc, _COUPLING_FIELD[t]) for loc, t in self._target_coupling_locs.items()),
        ]:
            if field not in self._rep_fields.setdefault(loc, []):
                self._rep_fields[loc].append(field)

        # One O(N) drop_duplicates gives the group combos (no per-obs logic). Encoding a group needs only its values
        # and the `.uns` tables, so it is deferred to batch time (cached), keyed by the group's id. An unconditional
        # schema (no groups, no conditions) has one group of every cell, ``()``, which scfit streams as is.
        uniq = obs_columns(adatas, self._group_cols).drop_duplicates()
        self._leaves: list[tuple] = list(map(tuple, uniq.to_numpy())) if self._group_cols else [()]
        self._leaf_to_gid: dict[tuple, int] = {leaf: i for i, leaf in enumerate(self._leaves)}
        self._adatas = adatas
        self._uns = adatas[0].uns  # the tables groups are encoded from; the same in every AnnData
        self._encode_cache: dict[int, dict[str, Any]] = {}

    @cached_property
    def _leaf_size(self) -> dict[tuple, int]:
        """Observations per group -- a full-obs groupby, so it is paid on first use rather than at construction.

        Only :class:`EvalLoader` ever asks (to size a metadata-only batch): training reads its row count off
        the delivered tensor. Computing it eagerly meant every training loader -- one per split -- swept the
        whole obs for a number it never read.

        ``groupby(sort=False)`` and the ``drop_duplicates`` behind :attr:`_leaves` both order by first
        appearance, so the counts zip straight onto the leaves (``strict`` makes a divergence loud).
        """
        sub = obs_columns(self._adatas, self._group_cols)
        if not self._group_cols:
            return {(): len(sub)}
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

        Both are selectors bolted onto ``group_by`` -- a split excludes observations, a pair id names the source
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
            if value is None or key == "n_obs":
                continue
            step_data[key] = (
                {k: self._conform(v) for k, v in value.items()} if isinstance(value, dict) else self._conform(value)
            )
        return step_data

    def _encode_group_cached(self, gid: int) -> dict[str, Any]:
        """The group's categorical encodings (groups and categorical conditions), ``(1, c, d)``, cached per group.

        Cached already conformed to ``dtype``/``device``; every batch of the group reuses the same tensors.
        """
        if gid not in self._encode_cache:
            entry = self._encode_group(self._leaves[gid])
            self._encode_cache[gid] = {
                k: _group_row(self._conform(_as_tensor(v), streamed=False)) for k, v in entry.items()
            }
        return self._encode_cache[gid]

    def _encode_group(self, leaf: tuple) -> dict[str, np.ndarray]:
        """Encode one group's values -- its group columns and categorical conditions -- to ``{key: vector}``."""
        entry: dict[str, np.ndarray] = {}
        values = pd.DataFrame([leaf], columns=list(self._group_cols))
        for data in (
            self._cond_schema.categorical_data(values, self._uns),
            self._groups_schema.categorical_data(values, self._uns),
        ):
            if data is not None:
                entry.update({k: _leaf_encoding(v) for k, v in data.extract_reps().mapping.items()})
        return entry

    def _batch_size(self, batch: dict, target_state: Any, n_rows: int) -> int:
        """Rows the batch has, which the source is aligned to: what the primary delivered, else ``n_rows``.

        Measuring the delivered tensor is authoritative -- it already accounts for the sampler's
        ``drop_last`` and for a group shorter than its cap. Only a metadata-only pass (``reps=()``) reads
        no observations at all, and there the caller knows the count outright.
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
                "a group was matched to zero control observations, so there is no source to flow from. Check that "
                "every group value present in the primary also has controls (matching is on the group "
                "columns, or on the pair id under `matched_keys`), or predict unpaired by passing "
                "`control_values_dict={}`."
            )
        import torch

        return source_state[torch.arange(batch_size, device=source_state.device) % n_source]

    def _source_coupling_dict(self, batch: dict) -> dict[str, torch.Tensor]:
        """Assembles the source coupling dictionary for a `StepData`."""
        if _CONTROL not in batch:
            return {}
        return {
            coupling_term: _as_tensor(batch[_CONTROL][coupling_loc])
            for coupling_loc, coupling_term in self._source_coupling_locs.items()
        }

    def _target_coupling_dict(self, batch: dict, has_state: bool) -> dict[str, torch.Tensor]:
        """Assembles the target coupling dictionary for a `StepData`."""
        if not has_state:
            return {}
        return {
            coupling_term: _as_tensor(batch[_PRIMARY][coupling_loc])
            for coupling_loc, coupling_term in self._target_coupling_locs.items()
        }

    def _step_data_from_batch(self, batch: dict, gid: int, *, has_state: bool, n_rows: int) -> StepData:
        """Assemble one scfit batch (+ its group id) into a ``StepData``.

        The categorical encodings (groups and categorical conditions) are cached per group, one row
        ``(1, c, d)``, in ``target_group_data``; continuous covariates are the per-obs aligned reps already in the
        batch, in ``target_condition_data``, and the source is row-aligned to the same count, ``n_obs``. ``has_state=False`` omits the target state (metadata-only prediction), and
        ``n_rows`` is the caller's row count for that case -- see :meth:`_batch_size`.
        """
        target_state = _as_tensor(batch[_PRIMARY][self._state_loc]) if has_state else None
        source_state = _as_tensor(batch[_CONTROL][self._state_loc]) if _CONTROL in batch else None
        batch_size = self._batch_size(batch, target_state, n_rows)
        source_state = self._align_source(source_state, batch_size)

        group = self._encode_group_cached(gid)
        # continuous covariates: per-obs aligned reps (present only when the stream carries them)
        condition = {
            key: _as_tensor(batch[_PRIMARY][loc]) for loc, key in self._cond_cont_locs.items() if loc in batch[_PRIMARY]
        }
        response = {
            key: _as_tensor(batch[_PRIMARY][loc]) for loc, key in self._resp_cont_locs.items() if loc in batch[_PRIMARY]
        }

        # Get coupling dictionaries
        src_couplings = self._source_coupling_dict(batch)
        tgt_couplings = self._target_coupling_dict(batch, has_state)

        step_data: dict[str, Any] = dict.fromkeys(_STEP_DATA_KEYS)
        step_data["target_state"] = target_state
        step_data["source_state"] = source_state
        step_data["target_condition_data"] = condition or None
        step_data["target_group_data"] = group or None
        step_data["target_response_data"] = response or None
        step_data["n_obs"] = batch_size
        # Update step data with coupling reps
        step_data.update(src_couplings)
        step_data.update(tgt_couplings)
        return self._conform_batch(step_data)  # type: ignore[return-value]

    def _control_link(
        self,
        control_weights: dict[tuple, float] | None,
        *,
        in_memory: bool = False,
    ) -> Stream | None:
        # Under fixed matching the pair column *is* the match: one id shared by a target group and the source
        # group it flows from, which is the only thing scfit's value-equality matching can key on. Otherwise a
        # target is matched to whatever shares its group columns.
        match_cols = self._pair_cols or tuple(self._groups_schema.groups)
        # The control stream carries the pair column so `match_on` can reach it; without one it groups on the
        # full group columns (what its weights' keys are).
        group_by = [*self._group_cols, *self._pair_cols]
        if control_weights:
            return Stream(
                _CONTROL,  # its own source: only the AnnData holding source cells (see `_sources_by_role`)
                group_by=group_by,
                reps=_merge_reps((self._state_loc,), self._source_coupling_locs),
                weights=control_weights,
                match_on=list(match_cols),
                in_memory=in_memory,
            )
        return None

    def _sources_by_role(self, primary: Stream, control: Stream | None) -> dict[str, list[AnnData]]:
        """Each stream's source: only the AnnData holding its cells, checked to carry its reps.

        A dataset split across files can keep each side's reps where its cells are (e.g. ``source_rep_quad`` only
        in the controls' file): the target stream reads the files holding targets, the control stream the files
        holding sources, and a file holding both is in both. The roles come from the streams' weights, so callers
        still pass one list of AnnData.
        """
        sources = {_PRIMARY: self._files_holding(primary, "target")}
        if control is not None:
            sources[_CONTROL] = self._files_holding(control, "source")
            self._check_shared_widths(sources)
        return sources

    def _check_shared_widths(self, sources: dict[str, list[AnnData]]) -> None:
        """The linear coupling term compares sources with targets directly, so one width across every file.

        Within a side scfit already requires it; this compares across the sides, which read separate sources.
        The quadratic terms are exempt (each side's is only compared within itself), and so is the state: whether
        it must match depends on the method -- flowing from the sources needs one space, generating from noise
        conditioned on an encoded source (GENOT) does not -- so the method checks it, not the loader.
        """
        shared = {loc for loc, t in self._source_coupling_locs.items() if t == _SOURCE_LIN_TERM}
        index = {id(a): i for i, a in enumerate(self._adatas)}
        for loc in shared:
            widths = {
                (role, index[id(a)]): int(get_rep(a, loc).shape[1])
                for role, files in (("source", sources[_CONTROL]), ("target", sources[_PRIMARY]))
                for a in files
            }
            if len(set(widths.values())) > 1:
                where = ", ".join(f"{w} wide in AnnData {i} ({role}s)" for (role, i), w in sorted(widths.items()))
                raise ValueError(
                    f"{loc} ({' / '.join(self._rep_fields.get(loc, ['rep']))}) is {where}. The linear coupling "
                    "term compares sources with targets directly, so it needs one width in every file; align the "
                    "files to it (one PCA fitted over all of them, or an embedding both sides share when their states "
                    "live in different spaces), or match across spaces with the quadratic terms alone "
                    "(`use_linear=False`)."
                )

    def _files_holding(self, stream: Stream, role: str) -> list[AnnData]:
        """The AnnData with at least one ``stream``-selected cell, each checked to carry the stream's reps."""
        if stream.weights is None:
            held = list(range(len(self._adatas)))
        else:
            selected = {leaf for leaf, w in stream.weights.items() if w > 0}
            held = [
                i
                for i, a in enumerate(self._adatas)
                if not selected.isdisjoint(
                    map(tuple, obs_columns(a, list(stream.group_by)).drop_duplicates().to_numpy())
                )
            ]
        if not held:
            raise ValueError(
                f"None of the {len(self._adatas)} AnnData holds a {role} cell the schema selects, so there is "
                f"nothing to stream for the {role} side. Check that the group and condition values in the "
                f"schema (e.g. `control_values_dict`) match the data's obs."
            )
        for i in held:
            missing = [loc for loc in stream.reps if not _has_rep(self._adatas[i], loc)]
            if missing:
                described = ", ".join(f"{loc} ({' / '.join(self._rep_fields.get(loc, ['rep']))})" for loc in missing)
                raise ValueError(
                    f"AnnData {i} of {len(self._adatas)} holds {role} cells but lacks {described}, which the "
                    f"{role} side reads. Add it to that AnnData; each file only needs the reps of the cells it holds "
                    f"(e.g. `source_rep_quad` only where the sources are)."
                )
        return [self._adatas[i] for i in held]

    @property
    def group_cols(self) -> tuple[str, ...]:
        """The ``group_by`` columns (groups then categorical conditions) -- the order of a ``leaf`` tuple."""
        return self._group_cols

    @property
    def cond_cont_keys(self) -> tuple[str, ...]:
        """Continuous condition covariate keys carried per-obs in ``target_condition_data``."""
        return tuple(self._cond_cont_locs.values())

    @property
    def resp_keys(self) -> tuple[str, ...]:
        """Continuous response covariate keys carried per-obs in ``target_response_data``."""
        return tuple(self._resp_cont_locs.values())


class Loader(_StepDataBridge):
    """Yields ``StepData`` batches for one population, backed by :class:`scfit.data.Loader`.

    :param adata: The annotated data object to stream (the whole thing -- selection is by weights).
    :type adata: class: `AnnData | Sequence[AnnData]`

    :param condition_schema: The condition schema: its columns join ``group_by``, its continuous covariates
        stream as reps, and it encodes each group's categorical conditions.
    :type condition_schema: class: `ConditionDataSchema`

    :param groups_schema: The groups schema: its columns lead ``group_by`` and are what a control is matched
        on, and it encodes each group's categorical groups.
    :type groups_schema: class: `GroupsDataSchema`

    :param coupling_schema: The optional coupling schema. When provided, its linear
        (``rep_lin``) and per-side quadratic (``source_rep_quad`` / ``target_rep_quad``)
        reps are streamed alongside the state and surfaced as the
        ``source_coupling_*`` / ``target_coupling_*`` keys of each emitted ``StepData``.
        Defaults to `None`, in which case no coupling data is streamed.
    :type coupling_schema: class: `CouplingDataSchema | None`

    :param sample_rep: The ``obsm`` key holding the state representation, or ``None`` for ``.X``.
    :type sample_rep: class: `str | None`

    :param response_covs: Continuous response covariate ``obsm`` keys, streamed per-obs into
        ``target_response_data``.
    :type response_covs: class: `Collection[str]`

    :param split_by: An ``.obs`` column prepended to the primary's ``group_by``, so a leaf is
        ``(split, *group_cols)`` and ``primary_weights`` select observations rather than whole groups. ``None``
        (the default) groups on the group columns alone.
    :type split_by: class: `str | None`

    :param primary_weights: scfit ``{leaf tuple: weight}`` selecting the target (perturbed) groups for
        this loader; ``None`` streams every group uniformly. Controls should be given weight 0 here. The
        tuples are ``(split, *group_cols)`` when ``split_by`` is set, ``group_cols`` otherwise.
    :type primary_weights: class: `dict | None`

    :param control_weights: scfit ``{group tuple: weight}`` selecting controls out of ``adata`` itself.
        ``None`` => unpaired.
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
        adata: AnnDataLike,
        *,
        condition_schema: ConditionDataSchema,
        groups_schema: GroupsDataSchema,
        coupling_schema: CouplingDataSchema | None = None,
        sample_rep: str | None = None,
        response_covs: Collection[str] = (),
        split_by: str | None = None,
        primary_weights: dict[tuple, float] | None = None,
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
            coupling_schema=coupling_schema,
            sample_rep=sample_rep,
            response_covs=response_covs,
            pair_key=pair_key,
            dtype=dtype,
            device=device,
        )
        self._rows_per_batch = batch_size
        sampler = SamplerParams(
            batch_size=batch_size,
            chunk_size=chunk_size,
            preload_nchunks=preload_nchunks if preload_nchunks is not None else batch_size // chunk_size,
        )

        # Merge representations to avoid duplicates.
        primary_reps = _merge_reps(
            (self._state_loc,),
            self._cond_cont_locs,
            self._resp_cont_locs,
            self._target_coupling_locs,
        )

        # With a split column the leaf is ``(split, *group_cols)``, so a weight of 0 excludes the *observations*
        # of another split rather than the whole group -- see `DataManager.get_dataloaders`. The split is
        # only a selector, so `_to_step_data` strips it back off before looking the group's encoding up.
        prefix_cols = (split_by,) if split_by is not None else ()
        self._n_prefix = len(prefix_cols)

        primary = Stream(
            _PRIMARY,
            group_by=[*prefix_cols, *self._group_cols, *self._pair_cols],
            reps=primary_reps,
            weights=primary_weights,
        )
        # Control link -- `in_memory` materializes just the selected control observations, a small pool re-drawn
        # every batch. The control stream never groups on the split: controls are shared across splits.
        control_stream = self._control_link(control_weights, in_memory=True)
        links = {_CONTROL: control_stream} if control_stream is not None else {}
        # `self._adatas`, not the argument: an unconditional schema streams shallow copies carrying the
        # implicit all-observations group column (see `with_derived_obs`).
        sources = self._sources_by_role(primary, control_stream)

        # Make the scfit loader finite (one epoch by default) so a plain pass terminates and is
        # re-iterable; the caller (e.g. the trainer) sets the training length via `set_n_iters`.
        self._loader = ScfitLoader(
            sources, primary=primary, links=links, seed=seed, to=to, preload_to_gpu=preload_to_gpu, sampler=sampler
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

    .. note::
        Unlike :class:`Loader`, this loader does **not** stream coupling terms.
        Coupling is a training-time alignment between source and target populations;
        at prediction and validation time there is nothing to align against, and the
        emitted ``StepData`` carries ``None`` for every ``*_coupling_*`` key.

        If you later need coupling at eval -- e.g. for a GW alignment diagnostic --
        add an explicit opt-in here (``stream_coupling: bool = False``) rather than
        accepting a bare ``coupling_schema``: the schema-only form invites the caller
        to pass one and silently get coupling behaviour they did not ask for.

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

    :param max_per_group: Per-group cap (``None`` = all observations, ``N`` = at most N, ``1`` = one representative
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
        adata: AnnDataLike,
        *,
        condition_schema: ConditionDataSchema,
        groups_schema: GroupsDataSchema,
        sample_rep: str | None = None,
        response_covs: Collection[str] = (),
        primary_weights: dict[tuple, float] | None = None,
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
        # No coupling reps: this loader never streams coupling (see class docstring).
        primary_reps = _merge_reps(
            state_reps,
            self._cond_cont_locs,
            self._resp_cont_locs,
        )

        primary = Stream(
            _PRIMARY,
            group_by=[*self._group_cols, *self._pair_cols],
            reps=primary_reps,
            weights=primary_weights,
        )
        # Control link (source), matched on the group columns -- controls always carry a state to flow from.
        control_stream = self._control_link(control_weights)
        links = {_CONTROL: control_stream} if control_stream is not None else {}
        sources = self._sources_by_role(primary, control_stream)

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
            yield (
                self._step_data_from_batch(batch, self._leaf_to_gid[leaf], has_state=self._has_state, n_rows=n_rows),
                leaf,
            )

    def __len__(self) -> int:
        """Number of groups (== number of batches) this pass yields."""
        return len(self._loader)
