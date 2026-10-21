from __future__ import annotations

import copy
import dataclasses
from collections.abc import Collection, Mapping
from typing import TYPE_CHECKING, Any, NamedTuple, TypedDict, Unpack

import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from scfit.data import obs_columns
from scfit.registry import Component, component

if TYPE_CHECKING:
    import torch

    from sckitflow.data._loader import EvalLoader, Loader

from sckitflow._types import TargetCovariatesEncodingId
from sckitflow.data._dims import DataDimensions
from sckitflow.data._group_encoders import GroupEncoderConfig
from sckitflow.data._utils import AnnDataLike, as_adatas, with_derived_obs_stacked
from sckitflow.data.containers._categorical import CategoricalData
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

__all__ = ["DataManagerConfig", "LoaderKwargs", "DataManager"]


def _plain(dims: DataDimensions) -> dict[str, Any]:
    """``dims`` without its feature names, which compare as an index rather than with ``==``."""
    return {k: v for k, v in vars(dims).items() if k != "feature_names"}


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

    adatas: list[AnnData]
    pair_key: str | None
    extras: list[tuple]
    keys: list[tuple]
    is_source: np.ndarray
    is_target: np.ndarray


class LoaderKwargs(TypedDict, total=False):
    """The :class:`~sckitflow.data._loader.Loader` options :meth:`DataManager.get_dataloaders` forwards.

    Declared here rather than beside :class:`~sckitflow.data._loader.Loader` so that typing a call site
    does not import the scfit/annbatch stack (~2s) that the loader module pulls in.

    Everything describing *which* observations a loader streams (``primary_weights``,
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


class DataManager:
    """The schema: everything sckitflow needs to know about an ``AnnData`` before it can read one.

    Its job is to answer, from configuration alone, three questions a batch cannot answer for itself --
    **what** each observation's state and covariates are (``sample_rep``, ``conditions``, ``groups``, the
    ``*_covs``), **which observations flow into which** (``control_values_dict``, or ``matched_keys`` for fixed
    pairs), and **which observations are held out** (the ``split_by`` column). The loaders then read only what
    the schema declared.

    Two boundaries follow from that, and both are enforced rather than documented-and-hoped:

    **One owner per question.** Anything the schema declares is declared once, here, not re-passed per call:
    :meth:`get_dataloaders` takes no ``split_by``, and the split has exactly one source -- a ``split_by``
    column it requires to be present. The split itself is an experiment's decision, made before the data
    reaches a run (e.g. by a :mod:`~sckitflow.data.splitters` splitter writing the column). A declared column
    that is missing raises, and an *undeclared* ``"split"`` column raises rather than
    quietly training on held-out observations. Only inference-time overrides are per call
    (:meth:`get_eval_loader`'s ``control_values_dict`` / ``matched_keys``), because predicting over
    different controls than were registered is the point of them.

    **Derived columns never touch the caller's object.** The ``matched_keys`` pair id is sckitflow's own
    bookkeeping, but they have to live in ``.obs`` because that is where scfit reads
    grouping from. It is written to a shallow copy that shares ``X`` / ``obsm`` / ``uns``
    (:func:`~sckitflow.data._utils.with_derived_obs`), so nothing is duplicated and the caller's AnnData
    gains no columns it did not ask for -- which also keeps a *view* from being silently materialized.

    Built from a :class:`DataManagerConfig`; see it for the individual options.
    """

    def __init__(self, config: DataManagerConfig | None = None) -> None:
        """Initializes the object.

        :param config: The schema; ``None`` declares nothing beyond ``.X`` as the state.
        """
        self.config = config = DataManagerConfig() if config is None else config
        self._control_values_dict = config.control_values_dict
        self._matched_keys = dict(config.matched_pairs) if config.matched_pairs else None
        self._condition_state_key = config.condition_state_key

        self._split_by = config.split_by

        self._state_data_schema = StateDataSchema(sample_rep=config.sample_rep)
        self._condition_data_schema = ConditionDataSchema(
            categorical_conditions=config.categorical_conditions,
            categorical_reps=config.categorical_reps,
            continuous_conditions=config.continuous_conditions,
        )
        self._coupling_data_schema = CouplingDataSchema(
            rep_lin=config.rep_lin,
            source_rep_quad=config.source_rep_quad,
            target_rep_quad=config.target_rep_quad,
            use_linear=config.use_linear,
        )
        self._target_data_schema = ResponseDataSchema(
            categorical_covs_dict=config.target_categorical_covs_dict,
            continuous_covs=config.target_continuous_covs,
        )
        self._groups_data_schema = GroupsDataSchema(
            groups=config.groups,
            groups_reps=config.groups_reps,
            groups_encoding=config.groups_encoding,
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

    @property
    def _training_loader_schema(self) -> dict[str, Any]:
        """Everything a *training* loader reads off this schema -- :attr:`_loader_schema`, plus coupling.

        Coupling is a training-time concern: it aligns source and target populations during the fit, and
        has nothing to align against at prediction or validation. It is therefore kept out of the shared
        :attr:`_loader_schema` (which :meth:`get_eval_loader` spreads) and added here for
        :meth:`get_dataloaders` alone.

        The split is the one place the asymmetry is *observable*: same schema, different dict, different
        loader. Keeping the two properties adjacent makes that visible at review time -- the temptation to
        fold coupling back into :attr:`_loader_schema` ("it's just one more key") is exactly what this
        separation exists to prevent.
        """
        return {**self._loader_schema, "coupling_schema": self._coupling_data_schema}

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
        *,
        sides: tuple[bool, bool] = (True, True),
    ) -> DistributionData:
        state_data: StateData | None = self._get_state_data(adata) if require_target_state else None
        condition_data: MixedTypeData = self._get_condition_data(adata)
        response_data: MixedTypeData = self._get_target_data(adata)
        groups_data: CategoricalData = self._get_groups_data(adata)
        if require_target_state:
            source, target = sides
            source_coupling_data, target_coupling_data = self._coupling_data_schema.get_sides(
                adata, source=source, target=target
            )
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
    ) -> DataDimensions:
        return DataDimensions.init_from_distribution_data(data, feature_names)

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
        adata: AnnDataLike,
    ) -> DataDimensions:
        """Registers the data dimensionalities from the input data according to the current schema.

        Several AnnData are read as one dataset (see :meth:`_parts`) and must agree on every dimension and on
        their feature names.

        :param adata: The annotated data object(s) which to extract the dimensionalities from.
        :type adata: class: `AnnData | Sequence[AnnData]`
        """
        parts = self._parts(adata)
        roles = self._roles(parts)
        dims = [self._dims_of(a, sides=r) for a, r in zip(parts, roles, strict=True)]
        # A side's quadratic dim comes from the files holding that side, and must agree across them.
        per_side = {"source_quad_dim": 0, "target_quad_dim": 1}
        merged = {}
        for field, side in per_side.items():
            widths = {i: getattr(d, field) for i, (d, r) in enumerate(zip(dims, roles, strict=True)) if r[side]}
            if len(set(widths.values())) > 1:
                raise ValueError(
                    f"{field} differs across the AnnData holding {('sources', 'targets')[side]}: "
                    + ", ".join(f"AnnData {i}: {w}" for i, w in widths.items())
                    + ". A side's quadratic rep needs one width in every file holding that side."
                )
            merged[field] = next(iter(widths.values()), None)
        # Everything else must agree across every file.
        base = _plain(dims[0])
        for i, other in enumerate(dims[1:], start=1):
            plain = _plain(other)
            differ = [f"{k} ({base[k]} vs {plain[k]})" for k in base if k not in per_side and plain[k] != base[k]]
            if not other.feature_names.equals(dims[0].feature_names):
                differ.append("feature_names")
            if differ:
                raise ValueError(f"AnnData {i} disagrees with AnnData 0 on {', '.join(differ)}.")
        return dataclasses.replace(dims[0], **merged)

    def _roles(self, parts: list[AnnData]) -> list[tuple[bool, bool]]:
        """Per AnnData, whether it holds sources and whether it holds targets, as the loaders route them."""
        if not self.group_cols:
            return [(False, True)] * len(parts)  # unconditional: every cell is a target, flowing from noise
        sel = self._selection(parts)
        n = len(self.group_cols)
        sources = {key[:n] for key, src in zip(sel.keys, sel.is_source, strict=True) if src}
        targets = {key[:n] for key, tgt in zip(sel.keys, sel.is_target, strict=True) if tgt}
        roles = []
        for part in parts:
            leaves = set(map(tuple, obs_columns(part, list(self.group_cols)).drop_duplicates().to_numpy()))
            roles.append((not leaves.isdisjoint(sources), not leaves.isdisjoint(targets)))
        return roles

    def _dims_of(self, adata: AnnData, *, sides: tuple[bool, bool] = (True, True)) -> DataDimensions:
        data: DistributionData = self._get_distribution_data(adata, sides=sides)
        n_features = data.state_data.X.shape[1]
        return self._get_data_dimensionalities(data, self._get_feature_names(adata, n_features))

    def _parts(self, adata: AnnDataLike) -> list[AnnData]:
        """``adata`` as a list read as one dataset, each part carrying the merged ``.uns`` tables the schema reads.

        A table may live in any of the parts: its entries are merged across them, so a part holding only
        controls need not repeat the condition tables. An entry in several parts must be the same in each, and
        every value the schema looks up must have an entry. The parts are shallow copies; the caller's AnnData
        are untouched.
        """
        adatas = as_adatas(adata)
        cond = self._condition_data_schema.categorical_conditions
        cols_of = {rep: list(cond[level]) for level, rep in (self.config.categorical_reps or {}).items()}
        for col, rep in (self.config.groups_reps or {}).items():
            cols_of.setdefault(rep, []).append(col)
        tables: dict[str, dict] = {}
        for key, cols in cols_of.items():
            table, owner = {}, {}
            for i, part in enumerate(adatas):
                for name, vector in part.uns.get(key, {}).items():
                    if name in table and not np.array_equal(table[name], vector):
                        raise ValueError(f".uns[{key!r}][{name!r}] differs between AnnData {owner[name]} and {i}.")
                    table.setdefault(name, vector)
                    owner.setdefault(name, i)
            if not table:
                raise KeyError(f"no AnnData has .uns[{key!r}], which the schema reads.")
            used = pd.unique(obs_columns(adatas, cols).to_numpy().ravel())
            if missing := sorted(str(v) for v in used if v not in table):
                raise KeyError(f".uns[{key!r}] has no entry for {missing}.")
            tables[key] = table
        if not tables:
            return adatas
        parts = []
        for part in adatas:
            local = copy.copy(part)
            local.uns = {**part.uns, **tables}
            parts.append(local)
        return parts

    def get_feature_names(self, adata: AnnDataLike) -> pd.Index:
        """Registers the feature names from the input data according to the current schema.

        :param adata: The annotated data object(s) which to extract the feature names from.
        :type adata: class: `AnnData | Sequence[AnnData]`
        """
        return self.get_data_dimensionalities(adata).feature_names

    def _is_control(self, uniq: pd.DataFrame, control_values_dict: dict[str, str] | None) -> np.ndarray:
        """Which rows of ``uniq`` are controls, per the control value of every condition level."""
        cvd = control_values_dict if control_values_dict is not None else self._control_values_dict
        conditions = self._condition_data_schema.categorical_conditions
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

    def _pair_column(self, obs: pd.DataFrame, ids: dict[tuple, str]) -> pd.Categorical:
        """Per-obs pair id: its group's id, or ``""`` for a group in no pair (never matched, never streamed).

        Factorizes the group columns once (O(N) at C level) and looks the id up per *group*, so the per-obs
        mapping never runs python per row.
        """
        codes, uniques = pd.factorize(pd.MultiIndex.from_frame(obs.loc[:, list(self.group_cols)]))
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
        adatas: list[AnnData],
        *,
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
        obs = obs_columns(adatas, [*extra_cols, *self.group_cols])
        uniq = obs.drop_duplicates()
        combos = [tuple(row) for row in uniq.loc[:, list(self.group_cols)].to_numpy()]
        extras = [tuple(row) for row in uniq.loc[:, list(extra_cols)].to_numpy()]

        matched_keys = self._matched_keys if matched_keys is None else matched_keys
        if not matched_keys:
            is_source = self._is_control(uniq, control_values_dict)
            # Without named pairs, every non-control combination is a target.
            return _Selection(adatas, None, extras, combos, is_source, ~is_source)

        ids = self._pair_ids(matched_keys)
        sources = {tuple(key) for key in matched_keys}
        targets = {tuple(key) for key in matched_keys.values()}
        # An unknown key is a hard error: streaming nothing for it is how a typo becomes a training run.
        self._assert_keys_present(targets, combos, "target")
        self._assert_keys_present(sources, combos, "source")

        adatas = with_derived_obs_stacked(adatas, {_PAIR_KEY: self._pair_column(obs, ids)})
        return _Selection(
            adatas,
            _PAIR_KEY,
            extras,
            [(*combo, ids.get(combo, "")) for combo in combos],
            np.array([combo in sources for combo in combos], dtype=bool),
            np.array([combo in targets for combo in combos], dtype=bool),
        )

    def _resolve_split(self, adatas: list[AnnData]) -> tuple[list[AnnData], str | None]:
        """The adata to stream and the column to split it by.

        A declared ``split_by`` is only checked -- missing is an error, since silently training on everything
        is the one outcome nobody wants.
        """
        if self._split_by is not None:
            for i, adata in enumerate(adatas):
                if self._split_by not in adata.obs.columns:
                    raise KeyError(
                        f"split_by={self._split_by!r} was declared on this DataManager but is not in the .obs of "
                        f"AnnData {i} (columns: {list(adata.obs.columns)}). Run the splitter that produces it first."
                    )
            return adatas, self._split_by
        # Neither declared: one loader over everything. Guard the previous default (`split_by="split"`), so
        # data that carries a split cannot be trained on wholesale just because the schema forgot to say so.
        if any(_DEFAULT_SPLIT_KEY in adata.obs.columns for adata in adatas):
            raise ValueError(
                f"adata.obs has a {_DEFAULT_SPLIT_KEY!r} column but this DataManager declares no "
                "`split_by`, so every observation -- held-out ones included -- would train. "
                f"Declare `split_by={_DEFAULT_SPLIT_KEY!r}` to use it, or drop the column to train on all."
            )
        return adatas, None

    def get_dataloaders(
        self,
        adata: AnnDataLike,
        **loader_kwargs: Unpack[LoaderKwargs],
    ) -> dict[str, Loader]:
        """Build one streaming data loader per split value, as declared by the schema.

        Which observations are held out is the schema's business, not this call's: the declared ``split_by``
        column is read (and required). Without one, everything trains under a single ``"train"`` loader.

        Selection is by scfit weights over scfit's own leaf factorization -- no subset copying. The whole
        ``adata`` is streamed; each split's primary weights pick out that split's perturbed groups
        (controls get weight 0). Controls are shared across splits and drawn from ``adata`` itself by weight;
        the loader holds just them in memory.

        The split column is part of the primary's ``group_by``, so a leaf is ``(split, *group_cols)`` and a
        split's weights exclude observations rather than whole groups. That matters when a group spans splits (a
        per-obs split rather than a per-combination one): weighting on the group columns alone would let
        the held-out loader stream training observations.

        Which group is a target and which its source comes from the schema's ``control_values_dict``, or from
        its ``matched_keys`` for fixed pairs (see :meth:`_selection`).

        :param adata: The annotated data object to stream, or several read as one dataset.
        :type adata: class: `AnnData | Sequence[AnnData]`

        :param loader_kwargs: Forwarded to each :class:`Loader`. See :class:`LoaderKwargs` for the
            accepted options.

        :returns: Mapping from each split value (that has perturbed groups) to its loader.
        :rtype: class: `dict[str, Loader]`
        """
        for key in self._training_loader_schema:
            if key in loader_kwargs:
                raise TypeError(
                    f"{key} is owned by the schema, not per call: it is built from "
                    "`DataManagerConfig.rep_lin` / `source_rep_quad` / `target_rep_quad` / `use_linear`. "
                    "Pass it there, not to `get_dataloaders`."
                )
        from sckitflow.data._loader import Loader

        adatas, split_by = self._resolve_split(self._parts(adata))

        if split_by is None:
            # No split at all: one loader over every perturbed group, keyed "train". With no schema to
            # group on there is nothing to weight either -- `None` streams the whole adata uniformly.
            sel = self._selection(adatas)
            if self.group_cols:
                weights = {key: 1.0 for key, tgt in zip(sel.keys, sel.is_target, strict=True) if tgt}
                controls = {key: 1.0 for key, src in zip(sel.keys, sel.is_source, strict=True) if src}
            else:
                weights, controls = None, None
            return {
                "train": Loader(
                    sel.adatas,
                    **self._training_loader_schema,
                    primary_weights=weights,
                    control_weights=controls,
                    pair_key=sel.pair_key,
                    **loader_kwargs,
                )
            }

        sel = self._selection(adatas, extra_cols=(split_by,))

        # Controls are shared across splits, so their weights carry no split (the control link does not group
        # on it). The primary's do, one leaf per (split, group).
        control_weights = {key for key, src in zip(sel.keys, sel.is_source, strict=True) if src}
        primary_weights: dict[str, dict[tuple, float]] = {}
        for extra, key, tgt in zip(sel.extras, sel.keys, sel.is_target, strict=True):
            if tgt:
                primary_weights.setdefault(str(extra[0]), {})[(*extra, *key)] = 1.0

        return {
            split_value: Loader(
                sel.adatas,
                **self._training_loader_schema,
                split_by=split_by,
                primary_weights=weights,
                control_weights=dict.fromkeys(control_weights, 1.0) or None,
                pair_key=sel.pair_key,
                **loader_kwargs,
            )
            for split_value, weights in primary_weights.items()
        }

    def get_eval_loader(
        self,
        adata: AnnDataLike,
        *,
        max_per_group: int | None = None,
        require_target_state: bool = True,
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

        :param adata: The annotated data object to predict over, or several read as one dataset.
        :type adata: class: `AnnData | Sequence[AnnData]`

        :param max_per_group: Per-group cap on observations: ``None`` = all, ``N`` = at most N, ``1`` =
            predict-once-per-condition (dedup). Defaults to ``None``.
        :type max_per_group: class: `int | None`

        :param require_target_state: Whether a target state representation is read from ``adata``. Set to
            ``False`` for metadata-only prediction (``StepData["target_state"]`` is ``None``). Defaults to ``True``.
        :type require_target_state: class: `bool`

        :param control_values_dict: Overrides the instance's control values for this call. Defaults to
            ``None`` (use the instance's); ``{}`` predicts unpaired.
        :type control_values_dict: class: `dict[str, str] | None`

        :param matched_keys: Overrides the instance's fixed ``{source key: target key}`` pairs for this call.
            Defaults to ``None`` (use the instance's).
        :type matched_keys: class: `Mapping[tuple, tuple] | None`

        :param subsample: How ``max_per_group`` picks observations: ``"head"``, ``"random"``, or a callable.
        :type subsample: class: `str`

        :param dtype: Torch dtype every emitted tensor is cast to; ``None`` leaves them as streamed.
        :type dtype: class: `torch.dtype | None`

        :param device: torch.types.Device every emitted tensor is moved to; ``None`` leaves them as streamed. Unlike
            training this moves rather than asserts -- the eval path reads rows directly, so its one copy per
            group is unavoidable.
        :type device: class: `str | None`
        """
        from sckitflow.data._loader import EvalLoader

        sel = self._selection(
            self._parts(adata),
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
            sel.adatas,
            **self._loader_schema,
            primary_weights=primary_weights,
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


@component("data_manager", builds=DataManager)
class DataManagerConfig(Component):
    """The schema :class:`DataManager` is built from; portable, so it travels inside a run's spec.

    What it expects of the ``AnnData``, and how controls are held, is described in ``docs/data_format.md``.
    """

    sample_rep: str | None = None
    """String identifier for the state representation. When provided, it should appear as key in
    `.obsm` attribute of annotated data objects. Otherwise, the representation will fall back to
    `.X`."""

    categorical_conditions: dict[str, tuple[str, ...]] | None = None
    """Mapping from each condition level to the corresponding columns, used to initialize the
    conditioning data schema."""

    categorical_reps: dict[str, str] | None = None
    """Mapping from each condition level to the corresponding representation, used to initialize the
    conditioning data schema."""

    continuous_conditions: tuple[str, ...] | None = None
    """Collection of continuous condition covariates, used to initialize the conditioning data
    schema."""

    control_values_dict: dict[str, str] | None = None
    """Dictionary mapping each condition level to the corresponding value used to indicate control
    observations. Pass `{}` at call time (`get_eval_loader`, `predict_adata`) to predict
    unpaired, ignoring the registered controls."""

    matched_pairs: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] | None = None
    """Fixed matching: `[(source group key, target group key), ...]` over `group_cols` values, naming the pairs
    outright instead of deriving them from `control_values_dict` (whose source is always the control condition
    sharing a target's group columns). Takes precedence over `control_values_dict`. A group may appear in at
    most one pair -- it carries a single pair id -- so chains (`a -> b`, `b -> c`) and two sources for one
    target are rejected. Pairs rather than a mapping, because JSON cannot key on tuples."""

    split_by: str | None = None
    """The `.obs` column holding the split labels, written before the data reaches a run (e.g. by a
    :mod:`~sckitflow.data.splitters` splitter). Its presence is checked when loaders are built. ``None``
    means there is no split and every non-control group trains."""

    condition_state_key: str | None = None
    """The key for the continuous condition covariates to be viewed as state when
    `view_on_condition_space` is `True`. This argument is ignored otherwise."""

    target_categorical_covs_dict: dict[str, TargetCovariatesEncodingId] | None = None
    """Mapping indicating the encoding used to transform categorical target covariates, used to
    initialize the target data schema."""

    target_continuous_covs: tuple[str, ...] | None = None
    """Collection of string identifiers for the continuous target covariates, used to initialize the
    target data schema."""

    groups: tuple[str, ...] | None = None
    """Collection of string identifiers for grouping columns, used to initialize the grouping data
    schema."""

    groups_reps: dict[str, str] | None = None
    """Mapping for pre-computed representations of grouping covariates, used to initialize the target
    data schema."""

    groups_encoding: dict[str, GroupEncoderConfig] | None = None
    """Mapping from each group column to a :class:`~sckitflow.data._group_encoders.GroupEncoderConfig`
    (e.g. ``OneHotEncoderConfig(categories=("A549", "K562"))``, ``AffineTransformerConfig(scale=2.0)``), used to
    initialize the grouping data schema. Encoders are frozen configs that build their fitted transformer on
    demand, and the categorical ones list their categories explicitly."""

    rep_lin: str | None = None
    """String identifier for the linear representation of the states, used when matching
    distributions over partially incomparable states; this key will hold the optionally-shared features
    and will be used to initialize the coupling data schema"""

    source_rep_quad: str | None = None
    """String identifier for the state quadratic representation of source states, used when matching
    distributions over incomparable spaces. Used to initialize the coupling data schema.
    This key will hold the features exclusively present in the source distribution.
    """

    target_rep_quad: str | None = None
    """String identifier for the state quadratic representation of target states, used when matching
    distributions over incomparable spaces. Used to initialize the coupling data schema.
    This key will hold the features exclusively present in the target distribution.
    """

    use_linear: bool = True
    """Whether the coupling has a shared linear term. `False` selects pure Gromov-Wasserstein:
    source and target are matched on their intra-domain geometry alone, with no shared axes.
    Requires both `source_rep_quad` and `target_rep_quad`, and is mutually exclusive with
    `rep_lin`. Defaults to `True` (a shared linear term, optionally fused with per-side
    quadratics)."""

    def build(self) -> DataManager:
        """The data manager."""
        return DataManager(self)
