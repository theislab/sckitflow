import numpy as np
import pandas as pd
import pytest

from sc_flow.data.containers import CategoricalData


class TestCategoricalData:
    def test_init_basic(self) -> None:
        df = pd.DataFrame(
            {
                "cell_type": ["A", "B", "A"],
                "batch": ["x", "x", "y"],
            }
        )

        cat = CategoricalData.from_pandas(df)

        assert isinstance(cat, CategoricalData)
        assert len(cat) == 3
        assert list(cat.ann_df.columns) == ["cell_type", "batch"]
        for col in cat.ann_df.columns:
            assert hasattr(cat.ann_df[col], "cat"), f"{col} should be Categorical"

    def test_init_with_repr_dict_and_encoders(self) -> None:
        df = pd.DataFrame({"cell_type": ["A", "B", "A"]})

        repr_dict = {
            "cell_type": np.array([[1, 0], [0, 1], [1, 0]]),
        }
        categorical_encoders = {
            "cell_type": object(),  # encoder class placeholder
        }

        cat = CategoricalData.from_pandas(
            ann_df=df,
            repr_dict=repr_dict,
            categorical_encoders=categorical_encoders,
        )

        assert cat.repr_dict is repr_dict
        assert cat.categorical_encoders is categorical_encoders

    def test_len(self) -> None:
        df = pd.DataFrame({"a": range(5)})
        cat = CategoricalData.from_pandas(df)

        assert len(cat) == 5

    @pytest.mark.parametrize("idxs", [slice(0, 2), np.array([0, 2])])
    def test_getitem(self, idxs) -> None:
        df = pd.DataFrame(
            {
                "cell_type": ["A", "B", "C"],
                "batch": ["x", "x", "y"],
            }
        )
        repr_dict = {"dummy": np.ones((3, 1))}
        encoders = {"dummy": object()}

        cat = CategoricalData.from_pandas(
            ann_df=df,
            repr_dict=repr_dict,
            categorical_encoders=encoders,
        )

        subset = cat[idxs]

        assert isinstance(subset, CategoricalData)
        assert subset is not cat
        assert len(subset) == len(df.iloc[idxs])

        pd.testing.assert_frame_equal(
            subset.ann_df, df.iloc[idxs], check_dtype=False, check_categorical=False,
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
        repr_dict = {"cell_type": np.zeros((2, 2))}
        encoders = {"cell_type": object()}

        cat = CategoricalData.from_pandas(
            ann_df=df,
            repr_dict=repr_dict,
            categorical_encoders=encoders,
        )

        rep = repr(cat)

        assert "CategoricalData" in rep
        assert "n_obs=2" in rep
        assert "n_vars=2" in rep
        assert "cell_type" in rep
        assert "batch" in rep
        assert "repr_dict_keys=['cell_type']" in rep
        assert "categorical_encoders_keys=['cell_type']" in rep
