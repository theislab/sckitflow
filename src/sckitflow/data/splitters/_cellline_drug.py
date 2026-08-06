from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData

from sckitflow.data.splitters._base import Splitter

__all__ = ["CellLineDrugSplitter"]


class CellLineDrugSplitter(Splitter):
    """Splits ``(cell line, drug)`` combinations into train/test, keeping every cell line in train.

    The unit of splitting is a unique ``(cell_line, drug)`` combination. Combinations are held
    out **per cell line**, so that every cell line keeps at least one combination in the training
    set -- a cell line is never pushed entirely into the held-out split. Control observations
    (``drug == control_value``) are always assigned to train, so the paired control distribution
    stays available for every trained cell line.

    The hold-out is stratified per cell line: for a cell line with ``k`` non-control combinations,
    ``floor(test_fraction * k)`` of them are moved to test (capped at ``k - 1`` so at least one
    always remains in train). A cell line with a single non-control combination therefore keeps it
    in train. Selection is deterministic given ``seed``.
    """

    def __init__(
        self,
        *,
        cell_line_key: str = "cell_line",
        drug_key: str = "drug",
        control_value: str | None = "control",
        test_fraction: float = 0.2,
        seed: int = 0,
        split_key: str = "split",
        train_label: str = "train",
        test_label: str = "test",
    ) -> None:
        """Initializes the splitter.

        :param cell_line_key: ``adata.obs`` column holding the cell line. Defaults to ``"cell_line"``.
        :type cell_line_key: class: `str`

        :param drug_key: ``adata.obs`` column holding the drug/perturbation. Defaults to ``"drug"``.
        :type drug_key: class: `str`

        :param control_value: Value of ``drug_key`` marking control observations, always kept in
            train. Pass ``None`` to treat every observation as perturbed. Defaults to ``"control"``.
        :type control_value: class: `str | None`

        :param test_fraction: Target fraction of each cell line's non-control combinations to hold
            out into the test split, in ``[0, 1)``. Defaults to ``0.2``.
        :type test_fraction: class: `float`

        :param seed: Seed for the deterministic per-cell-line hold-out choice. Defaults to ``0``.
        :type seed: class: `int`

        :param split_key: ``adata.obs`` column the split label is written to. Defaults to ``"split"``.
        :type split_key: class: `str`

        :param train_label: Label written for training observations. Defaults to ``"train"``.
        :type train_label: class: `str`

        :param test_label: Label written for held-out observations. Defaults to ``"test"``.
        :type test_label: class: `str`
        """
        super().__init__(split_key=split_key)
        if not 0.0 <= test_fraction < 1.0:
            raise ValueError(f"test_fraction must be in [0, 1), got {test_fraction}.")
        self._cell_line_key = cell_line_key
        self._drug_key = drug_key
        self._control_value = control_value
        self._test_fraction = test_fraction
        self._seed = seed
        self._train_label = train_label
        self._test_label = test_label

    def assign(self, adata: AnnData) -> pd.Series:
        """Assigns each observation to train or test (see the class docstring for the policy)."""
        obs = adata.obs
        for col in (self._cell_line_key, self._drug_key):
            if col not in obs.columns:
                raise KeyError(f"{col!r} not found in adata.obs (columns: {list(obs.columns)}).")

        cell_line = obs[self._cell_line_key].to_numpy().astype(str)
        drug = obs[self._drug_key].to_numpy().astype(str)
        is_control = (
            np.zeros(len(obs), dtype=bool)
            if self._control_value is None
            else drug == str(self._control_value)
        )
        is_perturbed = ~is_control

        # Held-out combinations, chosen per cell line so at least one combination stays in train.
        rng = np.random.default_rng(self._seed)
        pert_combos = (
            pd.DataFrame({"cell_line": cell_line[is_perturbed], "drug": drug[is_perturbed]})
            .drop_duplicates()
            .sort_values(["cell_line", "drug"], kind="stable")
        )
        test_combos: list[tuple[str, str]] = []
        for cl, grp in pert_combos.groupby("cell_line", sort=True):
            drugs = grp["drug"].to_numpy()
            n_test = min(int(np.floor(self._test_fraction * len(drugs))), len(drugs) - 1)
            if n_test > 0:
                chosen = rng.choice(len(drugs), size=n_test, replace=False)
                test_combos.extend((cl, drugs[i]) for i in np.sort(chosen))

        labels = np.full(len(obs), self._train_label, dtype=object)
        if test_combos:
            combo_index = pd.MultiIndex.from_arrays([cell_line, drug])
            is_test = combo_index.isin(test_combos) & is_perturbed
            labels[is_test] = self._test_label
        return pd.Series(labels, index=obs.index, name=self._split_key)
