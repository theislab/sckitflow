import numpy as np
import pandas as pd
import pytest

from sckitflow.data.containers import CategoricalData


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


class TestStoredRepresentationShape:
    """A stored representation is one vector per value; an ambiguous shape must fail, not be flattened."""

    @staticmethod
    def _reps(stored: np.ndarray) -> np.ndarray:
        cat = CategoricalData.from_pandas(
            pd.DataFrame({"drug": ["d0"]}),
            repr_dict={"drug": {"d0": stored}},
            categorical_reps_map={"drug": "drug"},
        )
        (rep,) = cat.extract_reps().mapping.values()
        return np.asarray(rep)

    @pytest.mark.parametrize("stored", [np.arange(5.0), np.arange(5.0).reshape(1, 5)])
    def test_both_vector_forms_are_accepted(self, stored):
        """`(d,)` and the equivalent row form `(1, d)` are both in use across the repo."""
        assert self._reps(stored).shape == (1, 1, 5)

    @pytest.mark.parametrize("bad", [(2, 3), (1, 2, 3), (3, 1)])
    def test_an_ambiguous_shape_is_refused(self, bad):
        """A `reshape(1, -1)` here would flatten `(2, 3)` to `(1, 6)` while `DimsRegistry` reads 3."""
        with pytest.raises(ValueError, match=r"must be one vector per value"):
            self._reps(np.zeros(bad))

    def test_values_of_a_realm_must_share_a_width(self):
        """Nothing upstream checks this: a leaf holds one value, so `np.stack` never sees the mismatch.

        `DataDimensionalitiesRegistry` reads a realm's width off whichever value comes first, so a
        disagreement silently builds the model for one width and feeds it another.
        """
        cat = CategoricalData.from_pandas(
            pd.DataFrame({"drug": ["d0"]}),
            repr_dict={"drug": {"d0": np.zeros(5), "d1": np.zeros(3)}},
            categorical_reps_map={"drug": "drug"},
        )
        with pytest.raises(ValueError, match=r"realm 'drug' have differing widths \[3, 5\]"):
            cat.extract_reps()

    def test_a_consistent_realm_still_works(self):
        cat = CategoricalData.from_pandas(
            pd.DataFrame({"drug": ["d0"]}),
            repr_dict={"drug": {"d0": np.zeros(5), "d1": np.zeros(5)}},
            categorical_reps_map={"drug": "drug"},
        )
        (rep,) = cat.extract_reps().mapping.values()
        assert np.asarray(rep).shape == (1, 1, 5)


class TestEncoderOutputShape:
    """Every encoder must stack to `(n_obs, n_cols, d)`; `LabelEncoder` is the 1-D odd one out."""

    @staticmethod
    def _reps(encoder_id: str, values: list[str]) -> np.ndarray:
        from sckitflow.data._group_encoders import GroupEncoderContext, as_group_encoder

        col = np.asarray(values)
        encoder = as_group_encoder(encoder_id).build(GroupEncoderContext(col))
        cat = CategoricalData.from_pandas(
            pd.DataFrame({"ko": col}),
            categorical_encoders={"ko": encoder},
            categorical_reps_map={"ko": "ko"},
        )
        (rep,) = cat.extract_reps().mapping.values()
        return np.asarray(rep)

    def test_label_encoder_keeps_a_width_axis(self):
        """`LabelEncoder.transform` returns `(n_obs,)`; without the restored axis this stacks to (n, 1).

        `DimsRegistry` reports dim 1 for a label encoder, so a rank-2 stack builds the model for a
        width the loader never delivers -- and `_leaf_encoding` rejects the rank outright.
        """
        assert self._reps("label", ["koA", "control", "koA"]).shape == (3, 1, 1)

    def test_one_hot_encoder_width_is_the_vocabulary(self):
        assert self._reps("one-hot", ["koA", "control", "koA"]).shape == (3, 1, 2)

    def test_label_encoding_does_not_warn(self):
        """A column-vector into `LabelEncoder` "works" but raises sklearn's `DataConversionWarning`."""
        import warnings

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            self._reps("label", ["koA", "control"])
        assert [w for w in record if "column-vector" in str(w.message)] == []

    def test_a_rank_3_encoder_output_is_refused(self):
        class RankThreeEncoder:
            def transform(self, X):
                return np.zeros((X.shape[0], 2, 2))

        cat = CategoricalData.from_pandas(
            pd.DataFrame({"ko": ["koA"]}),
            categorical_encoders={"ko": RankThreeEncoder()},
            categorical_reps_map={"ko": "ko"},
        )
        with pytest.raises(ValueError, match=r"rank-3 array"):
            cat.extract_reps()
