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

    def test_concat_collection_two_objects(self) -> None:
        """Concatenate two compatible CategoricalData objects."""
        df1 = pd.DataFrame({"cell_type": ["A", "B"], "batch": ["x", "x"]})
        df2 = pd.DataFrame({"cell_type": ["C", "D"], "batch": ["y", "y"]})

        encoders = {col: DummyEncoder() for col in df1.columns}
        cat1 = CategoricalData.from_pandas(df1, categorical_encoders=encoders)
        cat2 = CategoricalData.from_pandas(df2, categorical_encoders=encoders)

        concatenated = CategoricalData.concat_collection([cat1, cat2])

        assert len(concatenated) == 4
        pd.testing.assert_frame_equal(
            concatenated.ann_df.reset_index(drop=True),
            pd.concat([df1, df2], axis=0).reset_index(drop=True),
            check_categorical=False,
        )
        # Merged dictionaries should contain all keys from both objects
        assert concatenated.categorical_encoders.keys() == encoders.keys()
        # Since both used the same encoders, the dicts are identical; update results in same
        assert concatenated.categorical_reps_map == {"cell_type": "cell_type", "batch": "batch"}

    def test_concat_collection_three_objects(self) -> None:
        """Concatenate three compatible objects."""
        dfs = [
            pd.DataFrame({"a": ["x"], "b": [1]}),
            pd.DataFrame({"a": ["y"], "b": [2]}),
            pd.DataFrame({"a": ["z"], "b": [3]}),
        ]
        encoders = {col: DummyEncoder() for col in dfs[0].columns}
        cats = [CategoricalData.from_pandas(df, categorical_encoders=encoders) for df in dfs]

        concatenated = CategoricalData.concat_collection(cats)
        assert len(concatenated) == 3
        expected_df = pd.concat(dfs, axis=0).reset_index(drop=True)
        pd.testing.assert_frame_equal(
            concatenated.ann_df.reset_index(drop=True),
            expected_df,
            check_categorical=False,
        )

    def test_concat_collection_preserves_repr_dict_and_encoders(self) -> None:
        """Different objects may have different representation sources (repr_dict vs encoder)."""
        # First object uses repr_dict for 'cell_type'
        df1 = pd.DataFrame({"cell_type": ["A"], "batch": ["x"]})
        repr_dict = {"cell_type": {"A": np.array([1, 0])}}
        cat1 = CategoricalData.from_pandas(df1, repr_dict=repr_dict)

        # Second object uses encoder for 'cell_type' (no repr_dict entry)
        df2 = pd.DataFrame({"cell_type": ["B"], "batch": ["y"]})
        encoders = {"cell_type": DummyEncoder(), "batch": DummyEncoder()}
        cat2 = CategoricalData.from_pandas(df2, categorical_encoders=encoders)

        concatenated = CategoricalData.concat_collection([cat1, cat2])

        # Both representation sources should be merged
        assert "cell_type" in concatenated.repr_dict
        assert concatenated.repr_dict["cell_type"] == {"A": np.array([1, 0])}
        assert "cell_type" in concatenated.categorical_encoders
        assert concatenated.categorical_encoders["cell_type"] is encoders["cell_type"]
        assert "batch" in concatenated.categorical_encoders
        assert concatenated.categorical_encoders["batch"] is encoders["batch"]

        # categorical_reps_map should combine entries from both (keys are column names)
        expected_map = {"cell_type": "cell_type", "batch": "batch"}
        assert concatenated.categorical_reps_map == expected_map

    def test_concat_collection_mismatched_columns_raises(self) -> None:
        """Concatenating objects with different column sets should raise."""
        df1 = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
        df2 = pd.DataFrame({"col1": [3, 4], "col3": ["c", "d"]})  # different second column

        encoders = {col: DummyEncoder() for col in df1.columns}
        cat1 = CategoricalData.from_pandas(df1, categorical_encoders=encoders)
        # Need encoders for df2 columns as well
        encoders2 = {col: DummyEncoder() for col in df2.columns}
        cat2 = CategoricalData.from_pandas(df2, categorical_encoders=encoders2)

        with pytest.raises(ValueError):  # check_sequence_query_against_reference raises ValueError
            CategoricalData.concat_collection([cat1, cat2])

    def test_concat_collection_empty_collection_raises(self) -> None:
        """Concatenating an empty list should raise an appropriate error."""
        with pytest.raises(ValueError, match="at least one"):
            CategoricalData.concat_collection([])

    def test_concat_collection_single_object_returns_copy(self) -> None:
        """Concatenating a single object should return a new (but equivalent) object."""
        df = pd.DataFrame({"A": ["a"], "B": [1]})
        encoders = {col: DummyEncoder() for col in df.columns}
        cat = CategoricalData.from_pandas(df, categorical_encoders=encoders)

        concatenated = CategoricalData.concat_collection([cat])

        assert concatenated is not cat
        assert len(concatenated) == len(cat)
        pd.testing.assert_frame_equal(concatenated.ann_df, cat.ann_df, check_categorical=False)
        # Dictionaries should be shallow copies? In current implementation they are merged (updated),
        # which for a single object results in identical content but new dict objects.
        assert concatenated.categorical_encoders == cat.categorical_encoders
        assert concatenated.repr_dict == cat.repr_dict
