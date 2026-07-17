"""Obs-only ``compile_obs`` — labels → dagloader ``Scheme`` + ``condition_fn``.

Replaces the flat ``prepare_data`` blob with the composed schema objects
(:class:`StateDataSchema` / :class:`ConditionDataSchema` / :class:`GroupsDataSchema`)
and compiles them **off ``obs`` (+ the ``uns`` embedding tables) only — cells are never
read here; they are streamed later by dagloader. This mirrors cellflow's
``build_annbatch_training`` but the condition encoder is sc_flow's own
:class:`CategoricalData`.

Two condition mechanisms (see the design note):

* **leaf-level** categorical/combinatorial covariates → the returned ``condition_fn``
  (a per-leaf lookup, constant within a class-coherent batch);
* **per-cell** "paired" covariates → extra ``Node`` keys (streamed aligned to the state
  cells), *not* handled here — pass them as additional ``state`` reps.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from sc_flow.data.containers._categorical import CategoricalData
from sc_flow.data.schemas._condition_data_schema import ConditionDataSchema
from sc_flow.data.schemas._groups_data_schema import GroupsDataSchema
from sc_flow.data.schemas._state_data_schema import StateDataSchema

__all__ = ["compile_obs", "CompiledData"]

Leaf = tuple[Any, ...]
ConditionFn = Callable[[Leaf], dict[str, np.ndarray]]


@dataclass(frozen=True)
class CompiledData:
    """Result of :func:`compile_obs` — everything dagloader needs, built from labels."""

    scheme: Any  # dagloader.Scheme
    condition_fn: ConditionFn
    cols: tuple[str, ...]
    data_dim: int | None = None


def _sample_rep_to_key(sample_rep: str) -> str:
    """``sample_rep`` → dagloader rep key (``"X"`` or ``"obsm/<rep>"``)."""
    return "X" if sample_rep == "X" else f"obsm/{sample_rep}"


def compile_obs(
    adata: Any,
    *,
    state: StateDataSchema,
    condition: ConditionDataSchema,
    groups: GroupsDataSchema | None = None,
    control_key: str,
    split_covariates: Sequence[str] = (),
) -> CompiledData:
    """Compile the composed schemas into a dagloader ``Scheme`` + ``condition_fn`` from obs only.

    :param adata: Source with ``.obs`` (labels) and ``.uns`` (embedding tables). Cells
        (``.X`` / ``.obsm``) are NOT read here — dagloader streams them at train time.
    :param state: Which representation to stream (becomes the ``Node`` key).
    :param condition: The leaf-level (categorical/combinatorial) condition covariates.
    :param groups: Embedded *sample* covariates (require a rep/encoding); ``None`` = none.
    :param control_key: Boolean/0-1 obs column marking control observations.
    :param split_covariates: Matching-context columns → the ``Bind.common`` (not embedded).
    """
    from dagloader import Bind, Node, Scheme, uniform

    obs: pd.DataFrame = adata.obs
    uns = getattr(adata, "uns", {}) or {}

    cond_cols = list(condition.all_condition_cols)
    group_cols = list(groups.groups) if groups is not None else []
    # grouping columns: matching context (split) + condition + embedded sample covariates,
    # deduped, order-preserving (context first) — matches cellflow's `cols` ordering.
    cols = tuple(dict.fromkeys([*split_covariates, *cond_cols, *group_cols]))
    key = _sample_rep_to_key(state.sample_rep)

    repr_dict = {level: uns[rep] for level, rep in condition.conditions_reps.items()}
    reps_map = condition.categorical_reps_map

    cond_idx = [cols.index(c) for c in cond_cols]

    def condition_fn(leaf: Leaf) -> dict[str, np.ndarray]:
        row = pd.DataFrame([{c: leaf[i] for c, i in zip(cond_cols, cond_idx, strict=True)}])
        cat = CategoricalData.from_pandas(row, repr_dict=repr_dict, categorical_reps_map=reps_map)
        return {k: np.asarray(v, dtype=np.float32) for k, v in cat.extract_reps().mapping.items()}

    ctrl_flag = obs[control_key].to_numpy().astype(bool)
    pert = [tuple(r) for r in obs.loc[~ctrl_flag, list(cols)].drop_duplicates().to_numpy()]
    ctrl = [tuple(r) for r in obs.loc[ctrl_flag, list(cols)].drop_duplicates().to_numpy()]

    scheme = Scheme(
        sources={"data": adata},
        nodes={
            "pert": Node("data", cols, key, uniform(pert)),
            "ctrl": Node("data", cols, key, uniform(ctrl)),
        },
        root="pert",
        binds=(Bind("pert", "ctrl", common=tuple(split_covariates)),),
        seed=0,
    )
    return CompiledData(scheme=scheme, condition_fn=condition_fn, cols=cols)
