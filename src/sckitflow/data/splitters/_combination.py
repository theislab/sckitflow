from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd
from anndata import AnnData

from sckitflow._utils import check_sequence_query_against_reference
from sckitflow.data.splitters._base import Splitter

__all__ = ["CombinationSplitter"]


class CombinationSplitter(Splitter):
    """Hold out whole ``group_keys`` combinations, so a held-out combination is unseen at training time.

    **What is split.** Not cells but *combinations*: the unique values of ``group_keys`` together, e.g. each
    ``(cell_line, drug)`` pair. Every cell of a combination gets the same label, so a test combination never
    leaks a single cell into train.

    **What is protected.** ``always_train_keys`` is a subset of ``group_keys`` naming what must stay
    represented in train -- pass ``["cell_line"]`` and no cell line is ever held out entirely, only some of
    its drugs. Concretely, the combinations are grouped by their ``always_train_keys`` values (each group is
    a *stratum*), and a stratum of ``k`` combinations gives up ``floor(test_fraction * k)`` of them, never
    more than ``k - 1``.

    **What that implies.** Rounding down is per stratum, so a stratum with fewer than
    ``ceil(1 / test_fraction)`` combinations gives up none: at ``test_fraction=0.2``, a cell line with 4
    drugs keeps all 4. :meth:`assign` warns if that leaves no test split at all.

    Control rows (``control_key == control_value``) are labelled ``control_label`` and never take part -- they
    are the shared source population, not a split. The choice is deterministic given ``seed``.

    Example, ``group_keys=["cell_line", "drug"]`` and ``always_train_keys=["cell_line"]`` at
    ``test_fraction=0.5``: cell line A with drugs ``d0..d3`` gives up 2 of them to test and keeps 2 in train;
    cell line B with a single drug keeps it; every control row is labelled ``control``.
    """

    def __init__(
        self,
        *,
        group_keys: Sequence[str],
        always_train_keys: Sequence[str],
        control_key: str | None = None,
        control_value: str = "control",
        test_fraction: float = 0.2,
        seed: int = 0,
        split_key: str = "split",
        train_label: str = "train",
        test_label: str = "test",
        control_label: str = "control",
    ) -> None:
        """Initializes the splitter.

        :param group_keys: ``adata.obs`` columns whose unique combination is the unit of splitting.
        :param always_train_keys: subset of ``group_keys`` whose every value keeps >=1 combination in train.
        :param control_key: optional ``adata.obs`` column marking controls (never split). ``None`` = no controls.
        :param control_value: value of ``control_key`` marking a control row. Defaults to ``"control"``.
        :param test_fraction: target fraction of each stratum's combinations to hold out, in ``[0, 1)``.
        :param seed: seed for the deterministic per-stratum hold-out choice.
        :param split_key: ``adata.obs`` column the split label is written to.
        :param train_label: label written for training observations.
        :param test_label: label written for held-out observations.
        :param control_label: label written for control observations (not split members).
        """
        super().__init__(split_key=split_key)
        if not 0.0 <= test_fraction < 1.0:
            raise ValueError(f"test_fraction must be in [0, 1), got {test_fraction}.")
        self._group_keys = tuple(group_keys)
        self._always_train_keys = tuple(always_train_keys)
        if not self._group_keys:
            raise ValueError("group_keys must be non-empty.")
        check_sequence_query_against_reference(
            self._always_train_keys,
            self._group_keys,
            query_name="always_train_keys",
            reference_name="group_keys",
        )
        if test_fraction > 0 and set(self._always_train_keys) == set(self._group_keys):
            # Every stratum would then be a single combination, and the "keep >=1 in train" cap makes its
            # hold-out 0 -- so no test_fraction could ever hold anything out. A config mistake, not a split.
            raise ValueError(
                f"always_train_keys {self._always_train_keys} covers every group key, so each stratum is one "
                "combination and nothing can ever be held out. Drop a key from always_train_keys, or pass "
                "test_fraction=0 if no hold-out is intended."
            )
        self._control_key = control_key
        self._control_value = control_value
        self._test_fraction = test_fraction
        self._seed = seed
        self._train_label = train_label
        self._test_label = test_label
        self._control_label = control_label

    def assign(self, adata: AnnData) -> pd.Series:
        """Assigns each observation to train / test / control (see the class docstring for the policy)."""
        obs = adata.obs
        needed = (*self._group_keys, *((self._control_key,) if self._control_key else ()))
        for col in needed:
            if col not in obs.columns:
                raise KeyError(f"{col!r} not found in adata.obs (columns: {list(obs.columns)}).")

        is_control = (
            obs[self._control_key].astype(str).to_numpy() == str(self._control_value)
            if self._control_key
            else np.zeros(len(obs), dtype=bool)
        )
        gk, atk = list(self._group_keys), list(self._always_train_keys)
        combos = obs.loc[~is_control, gk].astype(str).drop_duplicates()

        # Hold out per stratum (each `always_train_keys` value), always leaving >=1 combination in train.
        rng = np.random.default_rng(self._seed)
        test_combos: list[tuple] = []
        largest_stratum = 0
        strata = combos.groupby(atk, sort=True) if atk else [(None, combos)]
        for _, grp in strata:
            rows = list(map(tuple, grp[gk].to_numpy()))
            largest_stratum = max(largest_stratum, len(rows))
            n_test = min(int(np.floor(self._test_fraction * len(rows))), len(rows) - 1)
            if n_test > 0:
                test_combos.extend(rows[i] for i in np.sort(rng.choice(len(rows), size=n_test, replace=False)))

        if self._test_fraction > 0 and not test_combos:
            # `floor(test_fraction * k)` rounds down to 0 for every small stratum, so a hold-out was asked for
            # and none happened. Silence here reads as "split done" and only surfaces much later, as training
            # with no validation set.
            warnings.warn(
                f"nothing was held out: with test_fraction={self._test_fraction} a stratum needs at least "
                f"{int(np.ceil(1 / self._test_fraction))} combinations before floor(test_fraction * k) reaches "
                f"1, and the largest stratum here has {largest_stratum}. Every non-control observation is "
                f"labelled {self._train_label!r}.",
                UserWarning,
                stacklevel=2,
            )

        labels = np.full(len(obs), self._train_label, dtype=object)
        if test_combos:
            combo_index = pd.MultiIndex.from_arrays([obs[c].astype(str).to_numpy() for c in gk])
            labels[combo_index.isin(test_combos) & ~is_control] = self._test_label
        labels[is_control] = self._control_label
        return pd.Series(labels, index=obs.index, name=self._split_key)
