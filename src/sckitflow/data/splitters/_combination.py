from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from anndata import AnnData

from sckitflow.data.splitters._base import Splitter

__all__ = ["CombinationSplitter"]


class CombinationSplitter(Splitter):
    """Split unique ``group_keys`` combinations into train/test, keeping every ``always_train_keys`` value seen.

    The unit of splitting is a unique combination of ``group_keys`` (e.g. ``(cell_line, drug)`` or
    ``(a, b, c)``). Hold-out is stratified by ``always_train_keys`` (a subset of ``group_keys``): within
    each of its value combinations, ``floor(test_fraction * k)`` of the ``k`` combinations move to test,
    capped at ``k - 1`` -- so every ``always_train_keys`` value keeps at least one combination in train and
    is never pushed entirely into the held-out split. Control rows (``control_key == control_value``) are
    labeled ``control_label`` and never split. Deterministic given ``seed``.
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
        if not set(self._always_train_keys) <= set(self._group_keys):
            raise ValueError(
                f"always_train_keys {self._always_train_keys} must be a subset of group_keys {self._group_keys}."
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
        strata = combos.groupby(atk, sort=True) if atk else [(None, combos)]
        for _, grp in strata:
            rows = list(map(tuple, grp[gk].to_numpy()))
            n_test = min(int(np.floor(self._test_fraction * len(rows))), len(rows) - 1)
            if n_test > 0:
                test_combos.extend(rows[i] for i in np.sort(rng.choice(len(rows), size=n_test, replace=False)))

        labels = np.full(len(obs), self._train_label, dtype=object)
        if test_combos:
            combo_index = pd.MultiIndex.from_arrays([obs[c].astype(str).to_numpy() for c in gk])
            labels[combo_index.isin(test_combos) & ~is_control] = self._test_label
        labels[is_control] = self._control_label
        return pd.Series(labels, index=obs.index, name=self._split_key)
