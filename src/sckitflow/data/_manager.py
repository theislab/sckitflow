from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import TYPE_CHECKING, Any, NamedTuple, TypedDict, Unpack

import numpy as np
import pandas as pd
from anndata import AnnData

if TYPE_CHECKING:
    import torch

    from sckitflow.data._loader import EvalLoader, Loader

from sckitflow._types import TargetCovariatesEncodingId
from sckitflow.data._dims_registry import DataDimensionalitiesRegistry
from sckitflow.data._group_encoders import GroupEncoder, GroupEncoderId
from sckitflow.data._utils import with_derived_obs
from sckitflow.data.containers._categorical import CategoricalData
from sckitflow.data.containers._coupling import CouplingData
from sckitflow.data.containers._distribution import DistributionData
from sckitflow.data.containers._mixed_type import MixedTypeData
from sckitflow.data.containers._state import StateData
from sckitflow.data.schemas import (
    ConditionDataSchema,
    CouplingDataSchema,
    GroupsDataSchema,
    ResponseDataSchema,
    StateDataSchema,
)
from sckitflow.data.splitters import Splitter

__all__ = ["DataManagerKwargs", "LoaderKwargs", "DataManager"]

# The split column a `Splitter` writes by default -- only used to catch an adata that carries a split the
# schema never declared (see `DataManager._resolve_split`).
_DEFAULT_SPLIT_KEY = "split"

# ponytail: the derived `.obs` column carrying the pair id behind `matched_keys` -- a shim. scfit matches a
# linked stream to the primary by value equality on a shared column, so an explicit {source: target} mapping
# is only expressible once both members of a pair carry one shared value. It is written to a *shallow copy*
# of the caller's AnnData (`with_derived_obs`), never their object. Delete this and `_pair_*` once scfit
# grows explicit leaf->leaf pairing (a code-level remap of the factorization it already owns, no column).
_PAIR_KEY = "_sckitflow_pair"


class _Selection(NamedTuple):
    """What to stream and how to weight it, per unique group combination. See :meth:`DataManager._selection`.

    ``keys[i]`` is combination ``i``'s scfit weight key -- its group values, plus the pair id under fixed
    matching -- and ``extras[i]`` the ``extra_cols`` values (e.g. the split) that prefix it on the primary.
    """

    adata: AnnData
    control_adata: AnnData | None
    pair_key: str | None
    extras: list[tuple]
    keys: list[tuple]
    is_source: np.ndarray
    is_target: np.ndarray


class LoaderKwargs(TypedDict, total=False):
    """The :class:`~sckitflow.data._loader.Loader` options :meth:`DataManager.get_dataloaders` forwards.

    Declared here rather than beside :class:`~sckitflow.data._loader.Loader` so that typing a call site
    does not import the scfit/annbatch stack (~2s) that the loader module pulls in.

    Everything describing *which* cells a loader streams (``primary_weights``, ``control_adata``,
    ``control_weights``) is derived by :meth:`DataManager.get_dataloaders` and so is deliberately absent:
    these are the knobs that survive being passed through.
    """

    to: str | None
    """scfit's batch backend. ``None`` (the default) keeps annbatch's native arrays -- numpy on the host,
    cupy on a GPU-resident read window -- which the loader then maps onto torch without copying."""

    dtype: torch.dtype | None
    """Torch dtype every emitted tensor is cast to. ``None`` leaves the streamed dtype alone."""

    device: str | None
    """Device the streamed batches must *already* be on -- an assertion, not a move (a mismatch raises
    rather than copying every batch). ``None`` skips the check."""

    seed: int
    """Seed for scfit's sampling schedule."""

    n_iters: int | None
    """Batches one pass yields. ``None`` = one epoch over the primary."""

    batch_size: int
    """Rows per emitted batch."""

    chunk_size: int
    """annbatch read-slice size; ``1`` reads per row, ``>1`` requires leaf-contiguous runs."""

    preload_nchunks: int | None
    """Chunks per annbatch read window. ``None`` = ``batch_size // chunk_size``."""

    preload_to_gpu: bool
    """Hand scfit/annbatch a GPU-resident read window, so batches arrive on the device rather than being
    copied there -- what ``device`` asserts, and on a GPU the only way to satisfy it (requires
    ``sckitflow[gpu]``: cupy, Linux/CUDA only)."""


class DataManagerKwargs(TypedDict, total=False):
    """Keyword arguments accepted by :class:`DataManager`."""

    sample_rep: str | None
    """String identifier for the state representation. When provided, it should appear as key in
    `.obsm` attribute of annotated data objects. Otherwise, the representation will fall back to
    `.X`. Defaults to `None`."""

    conditions: dict[str, Collection[str]] | None
    """Mapping from each condition level to the corresponding columns, used to initialize the
    conditioning data schema. Defaults to `None`."""

    conditions_reps: dict[str, str] | None
    """Mapping from each condition level to the corresponding representation, used to initialize the
    conditioning data schema. Defaults to `None`."""

    conditions_covariates: Collection[str] | None
    """Collection of continuous condition covariates, used to initialize the conditioning data
    schema. Defaults to `None`."""

    control_values_dict: dict[str, str] | None
    """Dictionary mapping each condition level to the corresponding value used to indicate control
    observations. Defaults to `None`. Pass `{}` at call time (`get_eval_loader`, `Model.predict`) to predict
    unpaired, ignoring the registered controls."""

    matched_keys: dict[tuple, tuple] | None
    """Fixed matching: `{source group key: target group key}` over `group_cols` values, naming the pairs
    outright instead of deriving them from `control_values_dict` (whose source is always the control condition
    sharing a target's group columns). Takes precedence over `control_values_dict`. A group may appear in at
    most one pair -- it carries a single pair id -- so chains (`a -> b`, `b -> c`) and two sources for one
    target are rejected. Defaults to `None`."""

    splitter: Splitter | None
    """A :class:`~sckitflow.data.splitters.Splitter` that derives the train/test labels itself, so building
    loaders never depends on a preprocessing step having written the column. It is applied to a shallow copy
    (see `data._utils.with_derived_obs`), so the caller's AnnData is left alone. Mutually exclusive with
    `split_by`. Defaults to `None`."""

    split_by: str | None
    """An existing `.obs` column holding the split labels, for data that was split elsewhere. Its presence is
    checked when loaders are built. Mutually exclusive with `splitter`. Defaults to `None`, in which case
    there is no split and every non-control group trains."""

    condition_state_key: str | None
    """The key for the continuous condition covariates to be viewed as state when
    `view_on_condition_space` is `True`. This argument is ignored otherwise. Defaults to `None`."""

    target_categorical_covs_dict: Mapping[str, TargetCovariatesEncodingId] | None
    """Mapping indicating the encoding used to transform categorical target covariates, used to
    initialize the target data schema. Defaults to `None`."""

    target_continuous_covs: Collection[str] | None
    """Collection of string identifiers for the continuous target covariates, used to initialize the
    target data schema. Defaults to `None`."""

    groups: Collection[str] | None
    """Collection of string identifiers for grouping columns, used to initialize the grouping data
    schema. Defaults to `None`."""

    groups_reps: dict[str, str] | None
    """Mapping for pre-computed representations of grouping covariates, used to initialize the target
    data schema. Defaults to `None`."""

    groups_encoding: dict[str, GroupEncoder | GroupEncoderId] | None
    """Mapping from each group column to a :class:`~sckitflow.data._group_encoders.GroupEncoder`
    (e.g. ``OneHot()``, ``Label()``, ``Affine(scale=2.0)``), used to initialize the grouping data
    schema. Encoders are serializable dataclasses that build their fitted transformer on demand. The
    string ids ``"label"`` / ``"one-hot"`` are accepted as shorthand for the parameter-free encoders.
    Defaults to `None`."""

    n_shared_dims: int | None
    """The number of shared dimensions to be considered when matching distributions over
    incomparable spaces, used to initialize the coupling data schema. Defaults to `None`."""

    source_rep: str | None
    """String identifier for the state representation of source states, used when matching
    distributions over incomparable spaces. Used to initialize the coupling data schema.
    Defaults to `None`."""


class DataManager:
    """The schema: everything sckitflow needs to know about an ``AnnData`` before it can read one.

    Its job is to answer, from configuration alone, three questions a batch cannot answer for itself --
    **what** each cell's state and covariates are (``sample_rep``, ``conditions``, ``groups``, the
    ``*_covs``), **which cells flow into which** (``control_values_dict``, or ``matched_keys`` for fixed
    pairs), and **which cells are held out** (``splitter`` or ``split_by``). The loaders then read only what
    the schema declared.

    Two boundaries follow from that, and both are enforced rather than documented-and-hoped:

    **One owner per question.** Anything the schema declares is declared once, here, not re-passed per call:
    :meth:`get_dataloaders` takes no ``split_by``, and the split has exactly one source -- a ``splitter``
    this class applies itself, or a ``split_by`` column it requires to be present. Passing both raises, a
    declared column that is missing raises, and an *undeclared* ``"split"`` column raises rather than
    quietly training on held-out cells. Only inference-time overrides are per call
    (:meth:`get_eval_loader`'s ``control_values_dict`` / ``matched_keys``), because predicting over
    different controls than were registered is the point of them.

    **Derived columns never touch the caller's object.** The split label and the ``matched_keys`` pair id
    are sckitflow's own bookkeeping, but they have to live in ``.obs`` because that is where scfit reads
    grouping from. They are written to a shallow copy that shares ``X`` / ``obsm`` / ``uns``
    (:func:`~sckitflow.data._utils.with_derived_obs`), so nothing is duplicated and the caller's AnnData
    gains no columns it did not ask for -- which also keeps a *view* from being silently materialized.

    See :class:`DataManagerKwargs` for the individual options.
    """

    def __init__(self, **kwargs: Unpack[DataManagerKwargs]) -> None:
        """Initializes the object. See :class:`DataManagerKwargs` for parameter descriptions."""
        self._control_values_dict = kwargs.get("control_values_dict")
        self._matched_keys = kwargs.get("matched_keys")
        self._condition_state_key = kwargs.get("condition_state_key")

        # One owner for the split: either sckitflow derives it (`splitter`) or the data already carries it
        # (`split_by`). Accepting both would mean two answers to "which cells are held out", decided by
        # whichever the loader consulted -- exactly the ambiguity this schema exists to remove.
        self._splitter: Splitter | None = kwargs.get("splitter")
        self._split_by: str | None = kwargs.get("split_by")
        if self._splitter is not None and self._split_by is not None:
            raise ValueError(
                f"pass either `splitter` (sckitflow derives the split) or `split_by` (an existing .obs "
                f"column), not both; got splitter={type(self._splitter).__name__} and "
                f"split_by={self._split_by!r}. The splitter writes {self._splitter.split_key!r}, so "
                "`split_by` is redundant with it."
            )

        self._state_data_schema = StateDataSchema(sample_rep=kwargs.get("sample_rep"))
        self._condition_data_schema = ConditionDataSchema(
            conditions=kwargs.get("conditions"),
            conditions_reps=kwargs.get("conditions_reps"),
            conditions_covariates=kwargs.get("conditions_covariates"),
        )
        self._coupling_data_schema = CouplingDataSchema(
            source_rep=kwargs.get("source_rep"),
            target_rep=kwargs.get("sample_rep"),
            n_shared_dims=kwargs.get("n_shared_dims"),
        )
        self._target_data_schema = ResponseDataSchema(
            categorical_covs_dict=kwargs.get("target_categorical_covs_dict"),
            continuous_covs=kwargs.get("target_continuous_covs"),
        )
        self._groups_data_schema = GroupsDataSchema(
            groups=kwargs.get("groups"),
            groups_reps=kwargs.get("groups_reps"),
            groups_encoding=kwargs.get("groups_encoding"),
        )

    @property
    def _view_on_condition_space(self) -> bool:
        return self._condition_state_key is not None

    @property
    def _loader_schema(self) -> dict[str, Any]:
        """Everything a loader reads off this schema -- handed over once, at construction.

        The loaders take this, not a ``DataManager``: what they need is the state rep, the covariate keys to
        stream, and the two schemas that encode a group (the only per-group question left at batch time).
        Nothing about splits, controls or pairs -- :meth:`get_dataloaders` / :meth:`get_eval_loader` have
        already turned those into weights by the time a loader exists.
        """
        return {
            "condition_schema": self._condition_data_schema,
            "groups_schema": self._groups_data_schema,
            "sample_rep": self._state_data_schema.sample_rep,
            "response_covs": self._target_data_schema.continuous_covs,
        }

    def _get_feature_names(
        self,
        adata: AnnData,
        n_features: int,
    ) -> pd.Index:
        """Determines feature names based on the state representation used."""
        if self._view_on_condition_space:
            sample_rep = self._condition_state_key
        else:
            sample_rep = self._state_data_schema.sample_rep

        # If we have a valid base name, build feature names as "base_0", "base_1", ...
        if sample_rep is not None:
            return pd.Index([f"{sample_rep}_{i}" for i in range(n_features)])

        # Fallback: use original var_names only if the number of features matches.
        if n_features == adata.shape[-1]:
            return adata.var_names

        # Last resort: generic feature names.
        return pd.Index([f"feature_{i}" for i in range(n_features)])

    def _get_state_data(
        self,
        adata: AnnData,
    ) -> StateData:
        return self._state_data_schema.get_data(adata)

    def _get_condition_data(
        self,
        adata: AnnData,
    ) -> MixedTypeData | None:
        return self._condition_data_schema.get_data(adata)

    def _get_coupling_data(self, adata: AnnData) -> tuple[CouplingData, CouplingData]:
        return self._coupling_data_schema.get_data(adata)

    def _get_groups_data(
        self,
        adata: AnnData,
    ) -> CategoricalData:
        return self._groups_data_schema.get_data(adata)

    def _get_target_data(
        self,
        adata: AnnData,
    ) -> MixedTypeData | None:
        return self._target_data_schema.get_data(adata)

    def _get_distribution_data(
        self,
        adata: AnnData,
        require_target_state: bool = True,
    ) -> DistributionData:
        state_data: StateData | None = self._get_state_data(adata) if require_target_state else None
        condition_data: MixedTypeData = self._get_condition_data(adata)
        response_data: MixedTypeData = self._get_target_data(adata)
        groups_data: CategoricalData = self._get_groups_data(adata)
        if require_target_state:
            source_coupling_data, target_coupling_data = self._get_coupling_data(adata)
        else:
            # coupling reps default to the state representation (`.X`) unless explicitly
            # configured otherwise, so they are unavailable in the same cases as state_data.
            source_coupling_data, target_coupling_data = None, None

        distribution_data = DistributionData(
            state_data,
            response_data=response_data,
            condition_data=condition_data,
            groups_data=groups_data,
            source_coupling_data=source_coupling_data,
            target_coupling_data=target_coupling_data,
        )

        if self._view_on_condition_space:
            return distribution_data.view_on_condition_space(self._condition_state_key)
        return distribution_data

    def _get_data_dimensionalities(
        self,
        data: DistributionData,
        feature_names: pd.Index,
    ) -> DataDimensionalitiesRegistry:
        return DataDimensionalitiesRegistry.init_from_distribution_data(data, feature_names)

    def get_distribution_data(
        self,
        adata: AnnData,
        require_target_state: bool = True,
    ) -> DistributionData:
        """Compiles an annotated data object into a distribution data container.

        :param adata: The annotated data object to compile.
        :type adata: class: `AnnData`

        :param require_target_state: Whether the state representation (`.X` or the configured
            `obsm` sample representation) is required from `adata`. Set to `False` to compile
            only the distribution metadata (condition/group covariates) when no target state
            data is available, e.g. for prediction without target states. Defaults to `True`.
        :type require_target_state: class: `bool`
        """
        return self._get_distribution_data(
            adata,
            require_target_state=require_target_state,
        )

    def get_data_dimensionalities(
        self,
        adata: AnnData,
    ) -> DataDimensionalitiesRegistry:
        """Registers the data dimensionalities from the input data according to the current schema.

        :param adata: The annotated data object which to extract the dimensionalities from.
        :type adata: class: `AnnData`
        """
        data: DistributionData = self._get_distribution_data(adata)
        n_features = data.state_data.X.shape[1]
        feature_names = self._get_feature_names(adata, n_features)
        return self._get_data_dimensionalities(data, feature_names)

    def get_feature_names(self, adata: AnnData) -> pd.Index:
        """Registers the feature names from the input data according to the current schema.

        :param adata: The annotated data object which to extract the feature names from.
        :type adata: class: `AnnData`
        """
        n_features = self._get_state_data(adata).X.shape[1]
        return self._get_feature_names(adata, n_features)

    def _combos(self, adata: AnnData) -> list[tuple]:
        """The unique ``group_cols`` combinations present in ``adata.obs``."""
        return [tuple(row) for row in adata.obs.loc[:, list(self.group_cols)].drop_duplicates().to_numpy()]

    def _is_control(self, uniq: pd.DataFrame, control_values_dict: dict[str, str] | None) -> np.ndarray:
        """Which rows of ``uniq`` are controls, per the control value of every condition level."""
        cvd = control_values_dict if control_values_dict is not None else self._control_values_dict
        conditions = self._condition_data_schema.conditions
        checks = [(col, str(val)) for level, val in (cvd or {}).items() for col in conditions[level]]
        if not checks:
            return np.zeros(len(uniq), dtype=bool)
        return np.logical_and.reduce([uniq[col].astype(str).to_numpy() == val for col, val in checks])

    def _pair_ids(self, matched_keys: Mapping[tuple, tuple]) -> dict[tuple, str]:
        """``{group combination: pair id}`` -- one id shared by a source group and its matched target.

        A group holds exactly one id, which rules out a group being in two pairs: a chain (``a -> b``,
        ``b -> c``) or two sources for one target. Both are rejected here rather than silently resolving to
        whichever pair happened to be assigned last. See :data:`_PAIR_KEY` for why an id exists at all.
        """
        ids: dict[tuple, str] = {}
        for i, (source_key, target_key) in enumerate(matched_keys.items()):
            for key in (tuple(source_key), tuple(target_key)):
                if key in ids:
                    raise ValueError(
                        f"group key {key} appears in more than one entry of matched_keys; a group carries a "
                        "single pair id, so it can be the source or the target of exactly one pair."
                    )
                ids[key] = str(i)
        return ids

    def _pair_column(self, adata: AnnData, ids: dict[tuple, str]) -> pd.Categorical:
        """Per-cell pair id: its group's id, or ``""`` for a group in no pair (never matched, never streamed).

        Factorizes the group columns once (O(N) at C level) and looks the id up per *group*, so the per-cell
        mapping never runs python per row.
        """
        codes, uniques = pd.factorize(pd.MultiIndex.from_frame(adata.obs.loc[:, list(self.group_cols)]))
        lut = np.array([ids.get(tuple(key), "") for key in uniques], dtype=object)
        return pd.Categorical(lut[codes])

    @staticmethod
    def _assert_keys_present(keys: Collection[tuple], combos: Collection[tuple], side: str) -> None:
        """Every ``matched_keys`` key must name a group combination that actually exists."""
        present = set(combos)
        missing = sorted(str(key) for key in keys if key not in present)
        if missing:
            raise KeyError(
                f"matched_keys {side} keys not found among the group combinations present: {missing}. Keys are "
                "tuples of `group_cols` values, in that order."
            )

    def _selection(
        self,
        adata: AnnData,
        *,
        control_adata: AnnData | None = None,
        extra_cols: Collection[str] = (),
        control_values_dict: dict[str, str] | None = None,
        matched_keys: Mapping[tuple, tuple] | None = None,
    ) -> _Selection:
        """Which unique combinations scfit streams as targets, which as sources, and under what weight key.

        Two matching modes, one shape. ``control_values_dict`` marks sources by their condition value, and a
        target's source is whatever shares its group columns. ``matched_keys`` names the pairs outright; each
        gets a pair id (:meth:`_pair_ids`) written to a derived column -- on a **shallow copy**, so the
        caller's AnnData is never touched -- which becomes both the trailing element of every weight key and
        the column the control link matches on.
        """
        uniq = adata.obs.loc[:, [*extra_cols, *self.group_cols]].drop_duplicates()
        combos = [tuple(row) for row in uniq.loc[:, list(self.group_cols)].to_numpy()]
        extras = [tuple(row) for row in uniq.loc[:, list(extra_cols)].to_numpy()]

        matched_keys = self._matched_keys if matched_keys is None else matched_keys
        if not matched_keys:
            is_source = self._is_control(uniq, control_values_dict)
            # Without named pairs, every non-control combination is a target.
            return _Selection(adata, control_adata, None, extras, combos, is_source, ~is_source)

        ids = self._pair_ids(matched_keys)
        sources = {tuple(key) for key in matched_keys}
        targets = {tuple(key) for key in matched_keys.values()}
        # Sources live in the control pool when there is one, in `adata` otherwise. An unknown key is a hard
        # error: streaming nothing for it is how a typo becomes a training run.
        self._assert_keys_present(targets, combos, "target")
        pool = self._combos(control_adata) if control_adata is not None else combos
        self._assert_keys_present(sources, pool, "source")

        adata = with_derived_obs(adata, **{_PAIR_KEY: self._pair_column(adata, ids)})
        if control_adata is not None:
            control_adata = with_derived_obs(control_adata, **{_PAIR_KEY: self._pair_column(control_adata, ids)})
        return _Selection(
            adata,
            control_adata,
            _PAIR_KEY,
            extras,
            [(*combo, ids.get(combo, "")) for combo in combos],
            np.array([combo in sources for combo in combos], dtype=bool),
            np.array([combo in targets for combo in combos], dtype=bool),
        )

    def _resolve_split(self, adata: AnnData) -> tuple[AnnData, str | None]:
        """The adata to stream and the column to split it by, per the schema's one declared owner.

        A ``splitter`` is applied here rather than expected to have been run: the split is part of the schema,
        so building loaders cannot depend on whether some upstream preprocessing wrote the column. It lands on
        a shallow copy, so the caller's AnnData never gains a column it did not ask for. A declared
        ``split_by`` is only checked -- missing is an error, since silently training on everything is the one
        outcome nobody wants.
        """
        if self._splitter is not None:
            return self._splitter.split(adata, copy=True), self._splitter.split_key
        if self._split_by is not None:
            if self._split_by not in adata.obs.columns:
                raise KeyError(
                    f"split_by={self._split_by!r} was declared on this DataManager but is not in adata.obs "
                    f"(columns: {list(adata.obs.columns)}). Run the splitter that produces it, or build the "
                    "DataManager with `splitter=` so sckitflow derives it."
                )
            return adata, self._split_by
        # Neither declared: one loader over everything. Guard the previous default (`split_by="split"`), so
        # data that carries a split cannot be trained on wholesale just because the schema forgot to say so.
        if _DEFAULT_SPLIT_KEY in adata.obs.columns:
            raise ValueError(
                f"adata.obs has a {_DEFAULT_SPLIT_KEY!r} column but this DataManager declares neither "
                "`split_by` nor `splitter`, so every observation -- held-out ones included -- would train. "
                f"Declare `split_by={_DEFAULT_SPLIT_KEY!r}` to use it, or drop the column to train on all."
            )
        return adata, None

    def get_dataloaders(
        self,
        adata: AnnData,
        *,
        control_adata: AnnData | None = None,
        **loader_kwargs: Unpack[LoaderKwargs],
    ) -> dict[str, Loader]:
        """Build one streaming data loader per split value, as declared by the schema.

        Which cells are held out is the schema's business, not this call's: the ``splitter`` given to
        :class:`DataManager` is applied here, or the declared ``split_by`` column is read (and required). With
        neither, everything trains under a single ``"train"`` loader.

        Selection is by scfit weights over scfit's own leaf factorization -- no subset copying. The whole
        ``adata`` is streamed; each split's primary weights pick out that split's perturbed groups
        (controls get weight 0). Controls are shared across splits: pass ``control_adata`` (a separate
        pool -- faster to load, and the cross-dataset case), or, when omitted, they are drawn from
        ``adata`` itself by weight.

        The split column is part of the primary's ``group_by``, so a leaf is ``(split, *group_cols)`` and a
        split's weights exclude cells rather than whole groups. That matters when a group spans splits (a
        per-cell split rather than a per-combination one): weighting on the group columns alone would let
        the held-out loader stream training cells.

        Which group is a target and which its source comes from the schema's ``control_values_dict``, or from
        its ``matched_keys`` for fixed pairs (see :meth:`_selection`).

        :param adata: The annotated data object to stream.
        :type adata: class: `AnnData`

        :param control_adata: Optional separate control (source) pool, shared by every split.
        :type control_adata: class: `AnnData | None`

        :param loader_kwargs: Forwarded to each :class:`Loader`. See :class:`LoaderKwargs` for the
            accepted options.

        :returns: Mapping from each split value (that has perturbed groups) to its loader.
        :rtype: class: `dict[str, Loader]`
        """
        from sckitflow.data._loader import Loader

        adata, split_by = self._resolve_split(adata)

        if split_by is None:
            # No split at all: one loader over every perturbed group, keyed "train". With no schema to
            # group on there is nothing to weight either -- `None` streams the whole adata uniformly.
            sel = self._selection(adata, control_adata=control_adata)
            if self.group_cols:
                weights = {key: 1.0 for key, tgt in zip(sel.keys, sel.is_target, strict=True) if tgt}
                controls = {key: 1.0 for key, src in zip(sel.keys, sel.is_source, strict=True) if src}
            else:
                weights, controls = None, None
            return {
                "train": Loader(
                    sel.adata,
                    **self._loader_schema,
                    primary_weights=weights,
                    control_adata=sel.control_adata,
                    control_weights=None if sel.control_adata is not None else controls,
                    pair_key=sel.pair_key,
                    **loader_kwargs,
                )
            }

        sel = self._selection(adata, control_adata=control_adata, extra_cols=(split_by,))

        # Controls are shared across splits, so their weights carry no split (the control link does not group
        # on it). The primary's do, one leaf per (split, group).
        control_weights = {key for key, src in zip(sel.keys, sel.is_source, strict=True) if src}
        primary_weights: dict[str, dict[tuple, float]] = {}
        for extra, key, tgt in zip(sel.extras, sel.keys, sel.is_target, strict=True):
            if tgt:
                primary_weights.setdefault(str(extra[0]), {})[(*extra, *key)] = 1.0

        return {
            split_value: Loader(
                sel.adata,
                **self._loader_schema,
                split_by=split_by,
                primary_weights=weights,
                control_adata=sel.control_adata,
                control_weights=(
                    None if sel.control_adata is not None else (dict.fromkeys(control_weights, 1.0) or None)
                ),
                pair_key=sel.pair_key,
                **loader_kwargs,
            )
            for split_value, weights in primary_weights.items()
        }

    def get_eval_loader(
        self,
        adata: AnnData,
        *,
        max_per_group: int | None = None,
        require_target_state: bool = True,
        control_adata: AnnData | None = None,
        control_values_dict: dict[str, str] | None = None,
        matched_keys: Mapping[tuple, tuple] | None = None,
        subsample: str = "head",
        to: str | None = None,
        dtype: torch.dtype | None = None,
        device: str | None = None,
        seed: int = 0,
    ) -> EvalLoader:
        """Build a deterministic, per-group :class:`~sckitflow.data._loader.EvalLoader` for prediction.

        Walks every perturbed group once (or a ``max_per_group`` cap), each matched to its control leaf,
        yielding ``(StepData, leaf)``. Controls are identified from ``control_values_dict``, or the pairs are
        named outright by ``matched_keys``; both arguments override the instance's, so inference is not bound
        to the controls or the pairs registered for training. With neither (``control_values_dict={}``) every
        group is predicted unpaired, with no source. Selection is by scfit weights over the whole ``adata``
        -- no subset copying.

        :param adata: The annotated data object to predict over.
        :type adata: class: `AnnData`

        :param max_per_group: Per-group cap on cells: ``None`` = all, ``N`` = at most N, ``1`` =
            predict-once-per-condition (dedup). Defaults to ``None``.
        :type max_per_group: class: `int | None`

        :param require_target_state: Whether a target state representation is read from ``adata``. Set to
            ``False`` for metadata-only prediction (``StepData["target_state"]`` is ``None``). Defaults to ``True``.
        :type require_target_state: class: `bool`

        :param control_adata: Optional separate control (source) pool, matched on the group columns.
        :type control_adata: class: `AnnData | None`

        :param control_values_dict: Overrides the instance's control values for this call. Defaults to
            ``None`` (use the instance's); ``{}`` predicts unpaired.
        :type control_values_dict: class: `dict[str, str] | None`

        :param matched_keys: Overrides the instance's fixed ``{source key: target key}`` pairs for this call.
            Defaults to ``None`` (use the instance's).
        :type matched_keys: class: `Mapping[tuple, tuple] | None`

        :param subsample: How ``max_per_group`` picks cells: ``"head"``, ``"random"``, or a callable.
        :type subsample: class: `str`

        :param dtype: Torch dtype every emitted tensor is cast to; ``None`` leaves them as streamed.
        :type dtype: class: `torch.dtype | None`

        :param device: Device every emitted tensor is moved to; ``None`` leaves them as streamed. Unlike
            training this moves rather than asserts -- the eval path reads rows directly, so its one copy per
            group is unavoidable.
        :type device: class: `str | None`
        """
        from sckitflow.data._loader import EvalLoader

        sel = self._selection(
            adata,
            control_adata=control_adata,
            control_values_dict=control_values_dict,
            matched_keys=matched_keys,
        )
        if self.group_cols:
            # Primary = target groups (every non-control group when unpaired); control link = the sources.
            primary_weights = {key: 1.0 for key, tgt in zip(sel.keys, sel.is_target, strict=True) if tgt}
            control_weights = {key: 1.0 for key, src in zip(sel.keys, sel.is_source, strict=True) if src}
        else:
            # Unconditional schema: one implicit group, nothing to select or pair against.
            primary_weights, control_weights = None, None
        return EvalLoader(
            sel.adata,
            **self._loader_schema,
            primary_weights=primary_weights,
            control_adata=sel.control_adata,
            control_weights=control_weights,
            pair_key=sel.pair_key,
            require_target_state=require_target_state,
            max_per_group=max_per_group,
            subsample=subsample,
            to=to,
            dtype=dtype,
            device=device,
            seed=seed,
        )

    @property
    def group_cols(self) -> tuple[str, ...]:
        """The columns a batch groups on: the group columns, then the categorical condition columns."""
        return (*tuple(self._groups_data_schema.groups), *self._condition_data_schema.all_condition_cols)

    @property
    def control_values_dict(self) -> dict[str, str] | None:
        """Exposes the homonymous attribute set at initialization."""
        return self._control_values_dict

    @property
    def matched_keys(self) -> dict[tuple, tuple] | None:
        """Exposes the homonymous attribute set at initialization."""
        return self._matched_keys

    @property
    def state_data_schema(self) -> StateDataSchema:
        """Exposes the state data schema."""
        return self._state_data_schema

    @property
    def condition_data_schema(self) -> ConditionDataSchema:
        """Exposes the condition data schema."""
        return self._condition_data_schema

    @property
    def coupling_data_schema(self) -> CouplingDataSchema:
        """Exposes the coupling data schema."""
        return self._coupling_data_schema

    @property
    def groups_data_schema(self) -> GroupsDataSchema:
        """Exposes the coupling data schema."""
        return self._groups_data_schema

    @property
    def target_data_schema(self) -> ResponseDataSchema:
        """Exposes the target data schema."""
        return self._target_data_schema

    @property
    def condition_state_key(self) -> str | None:
        """Returns the key used to define the condition covariates to be viewed as state."""
        return self._condition_state_key
