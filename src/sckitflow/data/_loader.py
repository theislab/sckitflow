r"""``Loader`` -- a training data loader that adapts :class:`scfit.data.Loader` to ``StepData``.

Selection is entirely scfit-native: one AnnData is streamed and the split + perturbed/control choice is
expressed as scfit **weights** over scfit's own leaf factorization (weight 0 == excluded, ``in_memory``
materializes only the selected cells). No Python-side masking or copying of subsets.

* the **primary** stream is the target/perturbed population -- controls get weight 0;
* the **control** link is the source population -- a separate ``control_adata`` (fast to load, and the
  cross-dataset case), or, when none is given, the *same* adata weighted so only controls are nonzero;
* **continuous** covariates ride as *aligned reps* (per-cell, straight from ``obsm``);
* **categorical** condition/group encodings are computed per group at batch time (cached), keyed off the
  group id scfit surfaces in ``batch["labels"]``, then tiled to the batch.

One scfit batch is one group, so each ``__next__`` yields one ``StepData``.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import numpy as np
from anndata import AnnData
from scfit.data import Loader as ScfitLoader, Stream

if TYPE_CHECKING:
    # StepData lives in ``core`` (which imports ``data``); import it for typing only to avoid a
    # runtime import cycle. At runtime we build the plain dict directly (see ``_STEP_DATA_KEYS``).
    from sckitflow.core._types import StepData
    from sckitflow.data._manager import DataManager

__all__ = ["Loader"]

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


class Loader:
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
        batch_size: int = 128,
        chunk_size: int = 1,
        preload_nchunks: int | None = None,
    ) -> None:
        self._dm = dm
        self._sampler_kwargs = {
            "batch_size": batch_size,
            "chunk_size": chunk_size,
            "preload_nchunks": preload_nchunks if preload_nchunks is not None else batch_size // chunk_size,
        }

        cond_schema = dm.condition_data_schema
        self._state_loc = _state_loc(dm.state_data_schema.sample_rep)
        # group_by = group columns + categorical condition columns (continuous covs are streamed reps).
        self._group_cols: tuple[str, ...] = (*tuple(dm.groups_data_schema.groups), *cond_schema.all_condition_cols)
        if not self._group_cols:
            raise ValueError(
                "SckitflowLoader needs at least one categorical group/condition column to group on "
                "(set `groups=` and/or `conditions=` on the DataManager)."
            )

        # Continuous covariates ride as aligned reps: {rep loc -> StepData dict key}.
        self._cond_cont_locs: dict[str, str] = {f"obsm/{c}": c for c in cond_schema.conditions_covariates}
        self._resp_cont_locs: dict[str, str] = {f"obsm/{c}": c for c in dm.target_data_schema.continuous_covs}
        primary_reps = (self._state_loc, *self._cond_cont_locs, *self._resp_cont_locs)

        # scfit surfaces the drawn group per batch via label_lookup; carry only a group id here and do the
        # (cached) categorical -> encoding at batch time (see `_to_step_data`), rather than up front.
        # One O(N) drop_duplicates gives the group combos + a representative row per group (no per-cell logic).
        uniq = adata.obs.loc[:, list(self._group_cols)].reset_index(drop=True).drop_duplicates()
        self._leaves: list[tuple] = list(map(tuple, uniq.to_numpy()))
        self._rep_idx: list[int] = uniq.index.to_numpy().tolist()
        self._adata = adata
        self._encode_cache: dict[int, tuple[dict[str, np.ndarray], dict[str, np.ndarray]]] = {}
        wanted = set(primary_weights) if primary_weights else set(self._leaves)
        label_lookup = {
            leaf: {"gid": np.array([i], dtype=np.int64)} for i, leaf in enumerate(self._leaves) if leaf in wanted
        }
        self._has_categorical = bool(label_lookup)

        primary = Stream(
            _PRIMARY,
            group_by=list(self._group_cols),
            reps=primary_reps,
            weights=primary_weights,
            label_lookup=label_lookup or None,
            **self._sampler_kwargs,
        )
        sources: dict[str, AnnData] = {_PRIMARY: adata}
        links: dict[str, Stream] = {}

        # Control link, matched on the group columns (e.g. cell line). Either a separate pool, or the same
        # adata weighted to controls-only -- scfit's in_memory materializes just those selected cells.
        match_cols = tuple(dm.groups_data_schema.groups)
        if control_adata is not None:
            sources[_CONTROL] = control_adata
            links[_CONTROL] = Stream(
                _CONTROL,
                group_by=list(match_cols) if match_cols else list(self._group_cols),
                reps=(self._state_loc,),
                match_on=list(match_cols),
                in_memory=True,
                **self._sampler_kwargs,
            )
        else:
            links[_CONTROL] = Stream(
                _PRIMARY,  # same source as the primary; weights pick out the controls
                group_by=list(self._group_cols),
                reps=(self._state_loc,),
                weights=control_weights,
                match_on=list(match_cols),
                in_memory=True,
                **self._sampler_kwargs,
            )
        self._paired = bool(links)
        self._loader = Loader(sources, primary=primary, links=links, seed=seed, to=to)

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

    def _to_step_data(self, batch: dict) -> StepData:
        """Assemble one scfit batch into a ``StepData`` (target/source state + conditioning).

        The categorical condition/group encoding is done here -- cached per group, keyed off the group id
        scfit surfaces in ``batch["labels"]`` -- and tiled to the batch. Continuous covariates are the
        per-cell aligned reps already in the batch.
        """
        target_state = _as_tensor(batch[_PRIMARY][self._state_loc])
        batch_size = int(target_state.shape[0])
        source_state = _as_tensor(batch[_CONTROL][self._state_loc]) if _CONTROL in batch else None

        cond_enc: dict[str, np.ndarray] = {}
        group_enc: dict[str, np.ndarray] = {}
        if self._has_categorical:
            cond_enc, group_enc = self._encode_group_cached(int(batch["labels"][_PRIMARY]["gid"][0]))
        condition: dict[str, Any] = {k: _tile(v, batch_size) for k, v in cond_enc.items()}
        group: dict[str, Any] = {k: _tile(v, batch_size) for k, v in group_enc.items()}
        # continuous covariates: per-cell aligned reps
        for loc, key in self._cond_cont_locs.items():
            condition[key] = _as_tensor(batch[_PRIMARY][loc])
        response = {key: _as_tensor(batch[_PRIMARY][loc]) for loc, key in self._resp_cont_locs.items()}

        step_data: dict[str, Any] = dict.fromkeys(_STEP_DATA_KEYS)
        step_data["target_state"] = target_state
        step_data["source_state"] = source_state
        step_data["target_condition_data"] = condition or None
        step_data["target_group_data"] = group or None
        step_data["target_response_data"] = response or None
        return step_data  # type: ignore[return-value]

    def __iter__(self) -> Iterator[StepData]:
        for batch in self._loader:
            yield self._to_step_data(batch)

    def __len__(self) -> int:
        """Number of batches in one epoch (as scfit computes it from the primary)."""
        return int(self._loader._n_batches)

    @property
    def loader(self) -> Loader:
        """The underlying :class:`scfit.data.Loader`."""
        return self._loader
