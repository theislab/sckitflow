from __future__ import annotations

import copy
import warnings
from typing import Self

import numpy as np
import pandas as pd
from anndata import AnnData
from pydantic import Field, model_validator
from scfit.registry import component

from sckitflow._utils import check_sequence_query_against_reference
from sckitflow.data.splitters._base import Splitter, SplitterConfig

__all__ = ["CombinationSplitter", "CombinationSplitterConfig"]


class CombinationSplitter(Splitter["CombinationSplitterConfig"]):
    """Hold out whole ``group_keys`` combinations, so a held-out combination is unseen at training time.

    **What is split.** Not observations but *combinations*: the unique values of ``group_keys`` together, e.g. each
    ``(cell_line, drug)`` pair. Every observation of a combination gets the same label, so a test combination
    never leaks a single row into train.

    **What is protected.** ``always_train_keys`` is a subset of ``group_keys`` naming what must stay
    represented in train -- pass ``["cell_line"]`` and no cell line is ever held out entirely, only some of
    its drugs. Concretely, the combinations are grouped by their ``always_train_keys`` values (each group is
    a *stratum*), and a stratum of ``k`` combinations gives up ``floor(test_fraction * k)`` of them, never
    more than ``k - 1``.

    **What that implies.** Rounding down is per stratum, so a stratum with fewer than
    ``ceil(1 / test_fraction)`` combinations gives up none: at ``test_fraction=0.2``, a cell line with 4
    drugs keeps all 4. :meth:`assign` warns if that leaves no test split at all.

    Control rows (``control_key == control_value``) are labelled ``control_label`` and never take part -- they
    are the shared source population, not a split. The choice is drawn from ``rng``, copied on each call, so repeated
    calls give the same split.

    Example, ``group_keys=["cell_line", "drug"]`` and ``always_train_keys=["cell_line"]`` at
    ``test_fraction=0.5``: cell line A with drugs ``d0..d3`` gives up 2 of them to test and keeps 2 in train;
    cell line B with a single drug keeps it; every control row is labelled ``control``.
    """

    def assign(self, adata: AnnData) -> pd.Series:
        """Assigns each observation to train / test / control (see the class docstring for the policy)."""
        obs = adata.obs
        needed = (*self.config.group_keys, *((self.config.control_key,) if self.config.control_key else ()))
        for col in needed:
            if col not in obs.columns:
                raise KeyError(f"{col!r} not found in adata.obs (columns: {list(obs.columns)}).")

        is_control = (
            obs[self.config.control_key].astype(str).to_numpy() == str(self.config.control_value)
            if self.config.control_key
            else np.zeros(len(obs), dtype=bool)
        )
        gk, atk = list(self.config.group_keys), list(self.config.always_train_keys)
        combos = obs.loc[~is_control, gk].astype(str).drop_duplicates()

        # Hold out per stratum (each `always_train_keys` value), always leaving >=1 combination in train.
        rng = copy.deepcopy(self._rng)
        test_combos: list[tuple] = []
        largest_stratum = 0
        strata = combos.groupby(atk, sort=True) if atk else [(None, combos)]
        for _, grp in strata:
            rows = list(map(tuple, grp[gk].to_numpy()))
            largest_stratum = max(largest_stratum, len(rows))
            n_test = min(int(np.floor(self.config.test_fraction * len(rows))), len(rows) - 1)
            if n_test > 0:
                test_combos.extend(rows[i] for i in np.sort(rng.choice(len(rows), size=n_test, replace=False)))

        if self.config.test_fraction > 0 and not test_combos:
            # `floor(test_fraction * k)` rounds down to 0 for every small stratum, so a hold-out was asked for
            # and none happened. Silence here reads as "split done" and only surfaces much later, as training
            # with no validation set.
            warnings.warn(
                f"nothing was held out: with test_fraction={self.config.test_fraction} a stratum needs at least "
                f"{int(np.ceil(1 / self.config.test_fraction))} combinations before floor(test_fraction * k) reaches "
                f"1, and the largest stratum here has {largest_stratum}. Every non-control observation is "
                f"labelled {self.config.train_label!r}.",
                UserWarning,
                stacklevel=2,
            )

        labels = np.full(len(obs), self.config.train_label, dtype=object)
        if test_combos:
            combo_index = pd.MultiIndex.from_arrays([obs[c].astype(str).to_numpy() for c in gk])
            labels[combo_index.isin(test_combos) & ~is_control] = self.config.test_label
        labels[is_control] = self.config.control_label
        return pd.Series(labels, index=obs.index, name=self._split_key)


@component("splitter.combination", builds=CombinationSplitter)
class CombinationSplitterConfig(SplitterConfig):
    """Holds out whole condition combinations. See :class:`CombinationSplitter`."""

    group_keys: tuple[str, ...] = Field(min_length=1)
    """``adata.obs`` columns whose unique combination is the unit of splitting."""
    always_train_keys: tuple[str, ...] = ()
    """Subset of ``group_keys`` for which every unique value keeps at least one combination in train."""
    control_key: str | None = None
    """``adata.obs`` column marking controls, which are never split. ``None`` means no controls."""
    control_value: str = "control"
    """Value of ``control_key`` marking a control row."""
    test_fraction: float = Field(default=0.2, ge=0.0, lt=1.0)
    """Target fraction of each stratum's combinations to hold out."""
    train_label: str = "train"
    test_label: str = "test"
    control_label: str = "control"
    """Label written for control observations, which are not split members."""

    @model_validator(mode="after")
    def _check(self) -> Self:
        check_sequence_query_against_reference(
            self.always_train_keys, self.group_keys, query_name="always_train_keys", reference_name="group_keys"
        )
        if self.test_fraction > 0 and set(self.always_train_keys) == set(self.group_keys):
            # Every stratum would then be one combination, and the "keep >=1 in train" cap makes its hold-out 0.
            raise ValueError(
                f"always_train_keys {self.always_train_keys} covers every group key, so each stratum is one "
                "combination and nothing can ever be held out. Drop a key from always_train_keys, or pass "
                "test_fraction=0 if no hold-out is intended."
            )
        return self

    def build(self, *, rng: np.random.Generator) -> CombinationSplitter:
        return CombinationSplitter(self, rng=rng)
