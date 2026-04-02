"""Shared fixtures for tests/data/.

One small AnnData built from the cartesian product of biological
vocabulary.  Both ``test_selector`` and ``test_datamanager`` pick
whichever columns they need -- the extra columns are simply ignored.
"""
import itertools

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------
CELL_LINES = ["HeLa", "HEK293", "Jurkat"]
DRUGS = ["control", "aspirin", "ibuprofen", "paclitaxel"]
DOSES = ["low", "medium", "high"]
BATCHES = ["batch_1", "batch_2", "batch_3"]
PLATES = ["plate_A", "plate_B", "plate_C"]
CELLS_PER_COMBINATION = 4

N_CELL_LINES = len(CELL_LINES)
N_DRUGS = len(DRUGS)
N_DOSES = len(DOSES)
N_BATCHES = len(BATCHES)
N_PLATES = len(PLATES)

N_GENES = 10
REP_DIM = 8

# Column order defines sort order: groups first, conditions second.
ALL_COLUMNS = {
    "batch": BATCHES,
    "cell_line": CELL_LINES,
    "plate": PLATES,
    "dose": DOSES,
    "drug": DRUGS,
}


def _make_obs(columns: dict[str, list[str]]) -> pd.DataFrame:
    """Sorted categorical obs from the cartesian product of *columns*.

    Each combination is repeated ``CELLS_PER_COMBINATION`` times so that
    leaf slices contain more than one row.
    """
    combos = list(itertools.product(*columns.values()))
    rows = [combo for combo in combos for _ in range(CELLS_PER_COMBINATION)]
    df = pd.DataFrame(rows, columns=list(columns.keys()))
    df = df.sort_values(list(columns.keys())).reset_index(drop=True)
    return df.astype("category")


@pytest.fixture()
def adata_small() -> AnnData:
    """A small deterministic AnnData with all vocabulary columns.

    obs has columns: batch, cell_line, plate, dose, drug
    .uns has a representation dict for every column.
    .X is a small random matrix (deterministic seed).
    """
    obs = _make_obs(ALL_COLUMNS)
    n_obs = len(obs)

    rng = np.random.default_rng(42)
    X = rng.standard_normal((n_obs, N_GENES)).astype(np.float32)

    uns: dict = {}
    for col, values in ALL_COLUMNS.items():
        uns[col] = {
            val: rng.standard_normal((1, REP_DIM)).astype(np.float32)
            for val in values
        }

    ad = AnnData(X=X, obs=obs)
    ad.uns = uns
    return ad
