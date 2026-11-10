"""The dimensions come from the reps' shapes and one group's encodings, so they never need the data itself."""

import dask
import dask.array as da
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from sckitflow.data import DataManagerConfig
from sckitflow.data._group_encoders import OneHotEncoderConfig


def _unreadable(n_obs: int, width: int) -> da.Array:
    """A rep whose shape is known but whose values raise if anything reads them."""

    def read():
        raise AssertionError("the dimensions read the data")

    return da.from_delayed(dask.delayed(read)(), shape=(n_obs, width), dtype=np.float32)


def test_the_dimensions_read_shapes_not_data():
    n = 20
    obs = pd.DataFrame(
        {"drug": ["control", "a", "b", "a"] * 5, "cell_line": ["x", "y"] * 10}, index=[str(i) for i in range(n)]
    )
    adata = AnnData(
        X=_unreadable(n, 30),
        obs=obs,
        obsm={"X_pca": _unreadable(n, 8), "X_quad": _unreadable(n, 3), "dose": _unreadable(n, 1)},
        uns={"drug": {d: np.ones(5) for d in ("control", "a", "b")}},
    )
    dims = (
        DataManagerConfig(
            target_sample_rep="obsm.X_pca",
            categorical_conditions={"drug": ("drug",)},
            categorical_reps={"drug": "drug"},
            continuous_conditions=("obsm.dose",),
            control_values_dict={"drug": "control"},
            groups=("cell_line",),
            groups_encoding={"cell_line": OneHotEncoderConfig(categories=("x", "y"))},
            rep_lin="obsm.X_pca",
            source_rep_quad="obsm.X_quad",
            target_rep_quad="obsm.X_quad",
        )
        .build()
        .get_data_dimensionalities(adata)
    )

    assert dims.state_dim == 8 and dims.source_lin_dim == dims.target_lin_dim == 8
    assert dims.source_quad_dim == dims.target_quad_dim == 3
    assert dims.condition_reps_dims == {"drug": 5} and dims.condition_continuous_dims == {"dose": 1}
    assert dims.groups_reps_dims == {"cell_line": 2}


def test_a_missing_rep_names_the_file_and_the_field():
    adata = AnnData(np.zeros((4, 3)), obs=pd.DataFrame(index=[str(i) for i in range(4)]))
    with pytest.raises(KeyError, match=r"AnnData 0 lacks A\.obsm\[.X_pca.\] \(`target_sample_rep`\)"):
        DataManagerConfig(target_sample_rep="obsm.X_pca").build().get_data_dimensionalities(adata)
