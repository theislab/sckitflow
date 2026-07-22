
from __future__ import annotations

import copy
import logging
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import anndata as ad
import numpy as np
import pandas as pd

from sc_flow.data._encoders import Encoder
from sc_flow.data.containers._categorical import CategoricalData
from sc_flow.data.schemas._condition_data_schema import ConditionDataSchema
from sc_flow.data.schemas._coupling_data_schema import CouplingDataSchema
from sc_flow.data.schemas._covariates_data_schema import CovariatesDataSchema
from sc_flow.data.schemas._state_data_schema import StateDataSchema

if TYPE_CHECKING:
    from annbatch import DatasetCollection

    # The cell source binded can stream from: in-memory / out-of-core / on-disk. Resolved by
    # ``binded._io.open_source``. Mirrors cellflow's ``DataInput``.
    DataInput = ad.AnnData | DatasetCollection | str | os.PathLike | Sequence[str | os.PathLike | ad.AnnData]

__all__ = ["compile_obs", "CompiledData", "CompiledDims"]

logger = logging.getLogger("sc_flow.data")

Leaf = tuple[Any, ...]
ConditionFn = Callable[[Leaf], dict[str, np.ndarray]]


@dataclass(frozen=True)
class CompiledDims:

    state: int  # feature dim of the streamed state rep (``obsm/<rep>`` or ``X``)
    condition: Mapping[str, int]  # condition realm -> encoding dim (trailing axis of condition_fn output)
    max_comb: int  # combination slots (the set/pool axis length) shared across realms
    coupling: Mapping[str, int] | None = None  # coupling role -> rep feature dim (only differing reps)


@dataclass(frozen=True)
class CompiledData:

    scheme: Any  # binded.Scheme
    condition_fn: ConditionFn
    cols: tuple[str, ...]
    coupling: dict[str, str] | None = None  # coupling role -> streamed rep loc-string (see compile_obs)
    dims: CompiledDims | None = None  # data dimensionalities (headers + encoders); None only if unset


def _sample_rep_to_key(sample_rep: str) -> str:
    return "X" if sample_rep == "X" else f"obsm/{sample_rep}"


def compile_obs(
    data: DataInput,
    *,
    state: StateDataSchema,
    condition: ConditionDataSchema,
    control_key: str,
    covariates: CovariatesDataSchema | None = None,
    coupling: CouplingDataSchema | None = None,
    match_context: Sequence[str] = (),
    rep_tables: Mapping[str, Mapping] | None = None,
    control_in_memory: bool = False,
    control_path: DataInput | None = None,
    min_runs_per_leaf: int = 0,
    seed: int = 0,
) -> CompiledData:
    from scfit.data import Bind, Node, Scheme, uniform
    from scfit.data._io import key_backings, obs_columns, open_source

    cond_cols = list(condition.all_condition_cols)
    cov_cols = list(covariates.covariates) if covariates is not None else []
    # grouping columns: matching context + condition + embedded covariates,
    # deduped, order-preserving (context first) — matches cellflow's `cols` ordering.
    cols = tuple(dict.fromkeys([*match_context, *cond_cols, *cov_cols]))
    key = _sample_rep_to_key(state.sample_rep)

    # --- coupling reps: stream aligned to source (ctrl) / target (pert) cells as extra Node keys,
    # only when the coupling space differs from the state rep (same-space matching reuses source/target).
    def _ref_loc(ref: Any) -> str:
        return Node("data", ("_ref_",), ref).keys[0]  # binded normalizes an accessor/str → loc-string

    coupling_locs = {role: _ref_loc(ref) for role, ref in coupling.refs.items()} if coupling is not None else {}
    tgt_extra = [coupling_locs[r] for r in ("tgt_lin", "tgt_quad") if r in coupling_locs]
    src_extra = [coupling_locs[r] for r in ("src_lin", "src_quad") if r in coupling_locs]
    pert_keys = tuple(dict.fromkeys([key, *tgt_extra]))
    ctrl_keys = tuple(dict.fromkeys([key, *src_extra]))

    # embedding lookup tables (explicit): in-memory AnnData contributes its `.uns`; a collection/path has
    # none, so a `lookup` there needs rep_tables passed (Lookup.fit raises otherwise — no store reads).
    if rep_tables is None:
        rep_tables = dict(data.uns) if isinstance(data, ad.AnnData) else {}

    # resolve the target source (AnnData / DatasetCollection / zarr path / list) and read ONLY the label
    # columns + streamed rep keys — cells are never materialized (mirrors build_annbatch_training). With a
    # separate control_path the target source serves only the perturbed reps; otherwise it also serves the
    # controls (both key sets) and needs the control_key column to split them.
    data_keys = list(dict.fromkeys(list(pert_keys) if control_path is not None else [*pert_keys, *ctrl_keys]))
    data_cols = list(cols) if control_path is not None else [*cols, control_key]
    source = open_source(data, keys=data_keys, cols=data_cols)
    obs = obs_columns(source, data_cols)

    def _fit_encoders(encoder_map: Mapping[str, Encoder], realm_to_cols: dict) -> dict[str, Encoder]:
        # Fit a COPY of each realm's encoder (never mutate the schema's objects): a lookup binds its
        # rep_tables entry, a data-fit encoder fits ONCE on the union of the realm's columns' values across
        # obs — so a per-leaf single value yields the full category-space encoding, not a dim-1 fit.
        fitted: dict[str, Encoder] = {}
        for realm, encoder in encoder_map.items():
            union = obs[list(realm_to_cols[realm])].to_numpy().reshape(-1)
            fitted[realm] = copy.deepcopy(encoder).fit(union, uns=rep_tables)
        return fitted

    # --- perturbation levels: a level's columns are its combination slots (stacked) ---
    cond_reps_map = condition.categorical_reps_map
    cond_encoders = _fit_encoders(condition.condition_encoders, condition.conditions)
    cond_idx = [cols.index(c) for c in cond_cols]

    # --- embedded covariates: one value per sample, tiled across the max_comb slots (cellflow's np.tile) ---
    cov_reps_map = {c: c for c in cov_cols}  # each covariate is its own realm
    cov_encoders = (
        _fit_encoders(covariates.covariate_encoders, {c: [c] for c in cov_cols}) if covariates is not None else {}
    )
    cov_idx = [cols.index(c) for c in cov_cols]

    # combination length = the (shared) perturbation-level column count; covariates tile to it.
    max_comb = max((len(v) for v in condition.conditions.values()), default=1)

    def condition_fn(leaf: Leaf) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        if cond_cols:
            row = pd.DataFrame([{c: leaf[i] for c, i in zip(cond_cols, cond_idx, strict=True)}])
            cat = CategoricalData.from_pandas(row, encoders=cond_encoders, categorical_reps_map=cond_reps_map)
            out.update({k: np.asarray(v, dtype=np.float32) for k, v in cat.extract_reps().mapping.items()})
        if cov_cols:
            srow = pd.DataFrame([{c: leaf[i] for c, i in zip(cov_cols, cov_idx, strict=True)}])
            scat = CategoricalData.from_pandas(srow, encoders=cov_encoders, categorical_reps_map=cov_reps_map)
            for cov, v in scat.extract_reps().mapping.items():
                v = np.asarray(v, dtype=np.float32)  # (1, 1, dim) — one covariate value
                if max_comb > v.shape[1]:
                    v = np.repeat(v, max_comb, axis=1)  # tile across the max_comb combination slots
                out[cov] = v
        return out

    # --- pert (target) leaves, and ctrl (source) leaves from wherever the controls live ---
    def _leaves(frame: pd.DataFrame, leaf_cols: Sequence[str]) -> list[Leaf]:
        return [tuple(r) for r in frame.loc[:, list(leaf_cols)].drop_duplicates().to_numpy()]

    def _target_leaves(frame: pd.DataFrame, leaf_cols: Sequence[str]) -> list[Leaf]:
        # Perturbed-only filter: drop any target leaf whose *shortest contiguous on-disk run* is smaller
        # than `min_runs_per_leaf` — not merely its total count. annbatch's chunk_size rule requires EVERY
        # run of a drawn leaf to be >= chunk_size, and a leaf fragmented across stores (e.g. one plate
        # contributing <chunk_size cells) has a short run even when its total is large; keying on total
        # would let it through and the loader would then reject chunk_size>1. On a single grouped store each
        # leaf is one run, so shortest-run == total and this matches the old behaviour. min_runs_per_leaf<=1
        # keeps every leaf (runs are >=1). First-occurrence order preserved.
        sub = frame.loc[:, list(leaf_cols)]
        n = len(sub)
        if n == 0 or min_runs_per_leaf <= 1:
            return [tuple(r) for r in sub.drop_duplicates().to_numpy()]
        # run boundaries: a new run starts wherever any leaf column differs from the previous row.
        changed = np.zeros(n, dtype=bool)
        changed[0] = True
        for c in leaf_cols:
            codes = pd.factorize(sub[c], use_na_sentinel=False)[0]
            changed[1:] |= codes[1:] != codes[:-1]
        starts = np.flatnonzero(changed)
        lengths = np.diff(np.append(starts, n))  # contiguous run length at each start
        min_run: dict[Leaf, int] = {}
        order: list[Leaf] = []  # first-occurrence order
        for row, length in zip(sub.to_numpy()[starts], lengths, strict=True):
            leaf = tuple(row)
            if leaf not in min_run:
                min_run[leaf] = int(length)
                order.append(leaf)
            elif length < min_run[leaf]:
                min_run[leaf] = int(length)
        kept = [leaf for leaf in order if min_run[leaf] >= min_runs_per_leaf]
        if (dropped := len(order) - len(kept)) > 0:
            logger.info("min_runs_per_leaf=%d dropped %d/%d target leaves (shortest on-disk run < threshold)",
                        min_runs_per_leaf, dropped, len(order))
        return kept

    sources: dict[str, Any] = {"data": source}
    if control_path is None:
        # controls live in `data`, split by control_key; optionally materialized in RAM.
        ctrl_flag = obs[control_key].to_numpy().astype(bool)
        pert = _target_leaves(obs.loc[~ctrl_flag], cols)
        ctrl = _leaves(obs.loc[ctrl_flag], cols)
        ctrl_node = Node("data", cols, ctrl_keys, uniform(ctrl), in_memory=control_in_memory)
    else:
        # controls come from a separate source, matched to targets purely by `match_context`.
        if not match_context:
            raise ValueError("control_path requires a non-empty match_context to bind controls to targets.")
        ctrl_cols = tuple(match_context)
        pert = _target_leaves(obs, cols)  # all of `data` are targets
        control_source = open_source(control_path, keys=list(ctrl_keys), cols=list(ctrl_cols))
        ctrl = _leaves(obs_columns(control_source, list(ctrl_cols)), ctrl_cols)
        sources["control"] = control_source
        ctrl_node = Node("control", ctrl_cols, ctrl_keys, uniform(ctrl), in_memory=control_in_memory)

    scheme = Scheme(
        sources=sources,
        nodes={"pert": Node("data", cols, pert_keys, uniform(pert)), "ctrl": ctrl_node},
        root="pert",
        binds=(Bind("pert", "ctrl", common=tuple(match_context)),),
        seed=seed,
    )

    # --- dimensionalities: array headers (no cells) + one condition_fn lookup (no sampler) ---
    # state/coupling reps come from `key_backings(...).shape[1]`; condition realm dims from running the
    # fitted condition_fn on one target leaf (label lookup only). Sizes the VF without touching a batch.
    state_dim = int(key_backings(source, key)[0].shape[1])
    cond_dims: dict[str, int] = {}
    if pert:
        cond_dims = {realm: int(arr.shape[-1]) for realm, arr in condition_fn(pert[0]).items()}
    coupling_dims = {role: int(key_backings(source, loc)[0].shape[1]) for role, loc in coupling_locs.items()}
    dims = CompiledDims(
        state=state_dim, condition=cond_dims, max_comb=int(max_comb), coupling=coupling_dims or None
    )

    return CompiledData(
        scheme=scheme, condition_fn=condition_fn, cols=cols, coupling=coupling_locs or None, dims=dims
    )
