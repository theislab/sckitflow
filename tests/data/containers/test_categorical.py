import numpy as np
import pandas as pd
import pytest

from sc_flow.data.containers import CategoricalData


# Dummy encoder that returns a fixed 2D array of ones (shape: (n_samples, 1))
class DummyEncoder:
    def transform(self, X):
        # X is (n_samples, 1) – we ignore its values and return a constant array
        n = X.shape[0]
        return np.ones((n, 1))


class TestCategoricalData:
    def test_init_basic(self) -> None:
        df = pd.DataFrame(
            {
                "cell_type": ["A", "B", "A"],
                "batch": ["x", "x", "y"],
            }
        )
        # Provide an encoder for each column (realm = column name)
        encoders = {col: DummyEncoder() for col in df.columns}

        cat = CategoricalData.from_pandas(df, categorical_encoders=encoders)

        assert isinstance(cat, CategoricalData)
        assert len(cat) == 3
        assert list(cat.ann_df.columns) == ["cell_type", "batch"]
        for col in cat.ann_df.columns:
            assert hasattr(cat.ann_df[col], "cat"), f"{col} should be Categorical"

    def test_init_with_repr_dict_and_encoders(self) -> None:
        df = pd.DataFrame({"cell_type": ["A", "B", "A"]})

        # repr_dict must map realm -> dict of value -> array
        repr_dict = {
            "cell_type": {
                "A": np.array([1, 0]),
                "B": np.array([0, 1]),
            }
        }
        # Provide an encoder for the same realm; it will be ignored because repr_dict takes precedence
        categorical_encoders = {"cell_type": DummyEncoder()}

        cat = CategoricalData.from_pandas(
            df,
            repr_dict=repr_dict,
            categorical_encoders=categorical_encoders,
        )

        assert cat.repr_dict is repr_dict
        assert cat.categorical_encoders is categorical_encoders
        # The realm "cell_type" is present in repr_dict, so no error

    def test_len(self) -> None:
        df = pd.DataFrame({"a": range(5)})
        encoders = {"a": DummyEncoder()}
        cat = CategoricalData.from_pandas(df, categorical_encoders=encoders)
        assert len(cat) == 5

    @pytest.mark.parametrize("idxs", [slice(0, 2), np.array([0, 2])])
    def test_getitem(self, idxs) -> None:
        df = pd.DataFrame(
            {
                "cell_type": ["A", "B", "C"],
                "batch": ["x", "x", "y"],
            }
        )
        # Provide encoders for both columns
        encoders = {col: DummyEncoder() for col in df.columns}
        # Also provide a dummy repr_dict for 'cell_type' (optional, just to test passing)
        repr_dict = {"cell_type": {"A": np.array([1, 0]), "B": np.array([0, 1]), "C": np.array([1, 1])}}

        cat = CategoricalData.from_pandas(
            df,
            repr_dict=repr_dict,
            categorical_encoders=encoders,
        )

        subset = cat[idxs]

        assert isinstance(subset, CategoricalData)
        assert subset is not cat
        assert len(subset) == len(df.iloc[idxs])

        pd.testing.assert_frame_equal(
            subset.ann_df,
            df.iloc[idxs],
            check_dtype=False,
            check_categorical=False,
        )

        # repr_dict and encoders are passed through unchanged
        assert subset.repr_dict is repr_dict
        assert subset.categorical_encoders is encoders

    def test_repr_contains_key_information(self) -> None:
        df = pd.DataFrame(
            {
                "cell_type": ["A", "B"],
                "batch": ["x", "y"],
            }
        )
        # Provide representations for both realms
        repr_dict = {"cell_type": {"A": np.array([1, 0]), "B": np.array([0, 1])}}
        encoders = {"batch": DummyEncoder()}

        cat = CategoricalData.from_pandas(
            df,
            repr_dict=repr_dict,
            categorical_encoders=encoders,
        )

        rep = repr(cat)

        assert "CategoricalData" in rep
        assert "n_obs=2" in rep
        assert "n_vars=2" in rep
        assert "cell_type" in rep
        assert "batch" in rep
        # repr_dict_keys includes 'cell_type', categorical_encoders_keys includes 'batch'
        assert "repr_dict_keys=['cell_type']" in rep
        assert "categorical_encoders_keys=['batch']" in rep
