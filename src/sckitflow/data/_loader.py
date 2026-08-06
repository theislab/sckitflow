r"""``SckitflowLoader`` -- a training data loader that adapts :class:`scfit.data.Loader` to ``StepData``.

The loader turns one AnnData (typically a single split) into scfit streams and yields ready
``StepData`` batches for the training loop. It reuses the :class:`~sckitflow.data.DataManager`
schema so the emitted conditioning keys match those the model was sized for.

Design (mirrors the split the ``DataManager`` schema already draws):

* the **primary** stream is the target/perturbed population, grouped by the group + categorical
  condition columns;
* an optional **control** link (present when ``control_values_dict`` is set) is the source
  population, matched to the primary on the group columns;
* **continuous** covariates (``conditions_covariates`` / target ``continuous_covs``) are streamed
  as *aligned reps* -- loaded per cell, row-aligned with the state matrix, straight from ``obsm``;
* **categorical** condition/group encodings are per-group and ride as scfit ``label_lookup`` (one
  vector per group), tiled to the batch on the way into ``StepData``.

One scfit batch corresponds to one group, so each ``__next__`` yields one ``StepData``.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import numpy as np
from anndata import AnnData
from scfit.data import Loader, Stream

if TYPE_CHECKING:
    # StepData lives in ``core`` (which imports ``data``); import it for typing only to avoid a
    # runtime import cycle. At runtime we build the plain dict directly (see ``_STEP_DATA_KEYS``).
    from sckitflow.core._types import StepData
    from sckitflow.data._manager import DataManager

__all__ = ["SckitflowLoader"]

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


class SckitflowLoader:
    """Yields ``StepData`` batches for one population, backed by :class:`scfit.data.Loader`.

    :param adata: The (already split) annotated data object to stream from.
    :type adata: class: `AnnData`

    :param dm: The data manager whose schema defines the state/condition/group layout.
    :type dm: class: `DataManager`

    :param to: scfit backend for the streamed reps -- ``"torch"`` (default) emits torch tensors.
    :type to: class: `str | None`

    :param seed: Seed for the reproducible scfit stream. Defaults to ``0``.
    :type seed: class: `int`

    :param batch_size: Rows per emitted batch (per group). Defaults to ``128``.
    :type batch_size: class: `int`

    :param chunk_size: annbatch read-slice size. Defaults to ``1`` (per-row reads, any layout).
    :type chunk_size: class: `int`

    :param preload_nchunks: annbatch read-window size; a positive multiple of
        ``batch_size // chunk_size``. Defaults to ``batch_size // chunk_size``.
    :type preload_nchunks: class: `int | None`
    """

    def __init__(
        self,
        adata: AnnData,
        *,
        dm: DataManager,
        to: str | None = "torch",
        seed: int = 0,
        batch_size: int = 128,
        chunk_size: int = 1,
        preload_nchunks: int | None = None,
    ) -> None:
        self._dm = dm
        self._to = to
        self._seed = seed
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

        # Split into target (perturbed) and, when paired, source (control) populations.
        control_mask = self._control_mask(adata)
        target_adata = adata[~control_mask].copy()
        self._paired = bool(control_mask.any())

        # Categorical condition/group encodings -> per-group label_lookup (reuses the schema so the
        # keys match the model's dims registry). Split key sets route labels to condition vs group.
        label_lookup, self._cond_cat_keys, self._group_keys = self._build_label_lookup(target_adata)

        primary = Stream(
            _PRIMARY,
            group_by=list(self._group_cols),
            reps=primary_reps,
            label_lookup=label_lookup or None,
            **self._sampler_kwargs,
        )
        sources: dict[str, AnnData] = {_PRIMARY: target_adata}
        links: dict[str, Stream] = {}
        if self._paired:
            group_cols = tuple(dm.groups_data_schema.groups) or self._group_cols
            sources[_CONTROL] = adata[control_mask].copy()
            links[_CONTROL] = Stream(
                _CONTROL,
                group_by=list(group_cols),
                reps=(self._state_loc,),
                match_on=list(group_cols),
                in_memory=True,
                **self._sampler_kwargs,
            )

        self._loader = Loader(sources, primary=primary, links=links, seed=seed, to=to)

    def _control_mask(self, adata: AnnData) -> np.ndarray:
        """Boolean mask of control (source) cells, from ``control_values_dict`` over condition columns."""
        n = adata.n_obs
        control_values = self._dm.control_values_dict
        if not control_values:
            return np.zeros(n, dtype=bool)
        conditions = self._dm.condition_data_schema.conditions
        mask = np.ones(n, dtype=bool)
        for level, value in control_values.items():
            for col in conditions[level]:
                mask &= adata.obs[col].astype(str).to_numpy() == str(value)
        return mask

    def _build_label_lookup(self, target_adata: AnnData) -> tuple[dict[tuple, dict[str, np.ndarray]], list, list]:
        """Per-group categorical encodings ``{group tuple: {key: vector}}`` plus the cond/group key sets.

        Categorical encodings are constant within a group, so they are extracted **per group**: the
        schema's ``extract_reps`` is a leaf-level op (it asserts a single category value), so each
        unique group is sliced out and encoded on its own. Reusing the schema keeps the emitted keys
        aligned with ``get_data_dimensionalities``.
        """
        obs = target_adata.obs
        group_frame = obs.loc[:, list(self._group_cols)]
        cond_keys: list[str] = []
        group_keys: list[str] = []
        lookup: dict[tuple, dict[str, np.ndarray]] = {}

        for _, row in group_frame.drop_duplicates().iterrows():
            key = tuple(row[c] for c in self._group_cols)
            mask = np.ones(obs.shape[0], dtype=bool)
            for col in self._group_cols:
                mask &= obs[col].to_numpy() == row[col]
            leaf = target_adata[mask]

            entry: dict[str, np.ndarray] = {}
            condition_data = self._dm.condition_data_schema.get_data(leaf)
            if condition_data is not None and condition_data.categorical_covariates is not None:
                reps = condition_data.categorical_covariates.extract_reps().mapping
                entry.update({k: _leaf_vector(v) for k, v in reps.items()})
                cond_keys = list(reps)
            groups_data = self._dm.groups_data_schema.get_data(leaf)
            if groups_data is not None:
                reps = groups_data.extract_reps().mapping
                entry.update({k: _leaf_vector(v) for k, v in reps.items()})
                group_keys = list(reps)
            lookup[key] = entry
        return lookup, cond_keys, group_keys

    def _to_step_data(self, batch: dict) -> StepData:
        """Assemble one scfit batch into a ``StepData`` (target/source state + conditioning)."""
        target_state = _as_tensor(batch[_PRIMARY][self._state_loc])
        batch_size = int(target_state.shape[0])
        source_state = _as_tensor(batch[_CONTROL][self._state_loc]) if _CONTROL in batch else None

        labels = batch.get("labels", {}).get(_PRIMARY, {})
        condition: dict[str, Any] = {k: _tile(labels[k], batch_size) for k in self._cond_cat_keys if k in labels}
        group: dict[str, Any] = {k: _tile(labels[k], batch_size) for k in self._group_keys if k in labels}
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
