from collections.abc import Collection

import numpy as np
import pandas as pd
import pytest

from sckitflow.data.containers._base import BaseData


class DummyData(BaseData):
    def __init__(self, X: np.ndarray):
        self.X = X

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx):
        return DummyData(self.X[idx])

    @classmethod
    def concat_collection(cls, collection: Collection["DummyData"]) -> "DummyData":
        """Concatenate DummyData objects vertically."""
        if not collection:
            raise ValueError("Cannot concatenate empty collection")
        X_list = []
        ref_dim = None
        for elem in collection:
            if ref_dim is None:
                ref_dim = elem.X.shape[1] if elem.X.ndim > 1 else 1  # handle 1D
            else:
                dim = elem.X.shape[1] if elem.X.ndim > 1 else 1
                if dim != ref_dim:
                    raise ValueError("Mismatched number of columns")
            X_list.append(elem.X)
        X_concat = np.concatenate(X_list, axis=0)
        return cls(X_concat)


class TestBaseData:
    def test_get_query_idxs_valid(self) -> None:
        ref = pd.MultiIndex.from_tuples(
            [("a", 1), ("a", 2), ("b", 1)],
            names=["x", "y"],
        )
        qry = pd.MultiIndex.from_tuples(
            [("a", 2), ("b", 1)],
            names=["x", "y"],
        )

        idxs = BaseData._get_query_idxs(ref, qry)

        np.testing.assert_array_equal(idxs, np.array([1, 2]))

    def test_get_query_idxs_non_unique_reference_raises(self) -> None:
        ref = pd.MultiIndex.from_tuples([("a", 1), ("a", 1)])
        qry = pd.MultiIndex.from_tuples([("a", 1)])

        with pytest.raises(ValueError, match="Reference index must be unique"):
            BaseData._get_query_idxs(ref, qry)

    def test_get_query_idxs_non_unique_query_raises(self) -> None:
        ref = pd.MultiIndex.from_tuples([("a", 1), ("b", 2)])
        qry = pd.MultiIndex.from_tuples([("a", 1), ("a", 1)])

        with pytest.raises(ValueError, match="Query index must be unique"):
            BaseData._get_query_idxs(ref, qry)

    def testassert_same_len_valid(self) -> None:
        a = DummyData(np.zeros((5, 2)))
        b = DummyData(np.ones((5, 3)))

        a.assert_same_len(b)

    def testassert_same_len_invalid(self) -> None:
        a = DummyData(np.zeros((5, 2)))
        b = DummyData(np.ones((4, 2)))

        with pytest.raises(ValueError, match="same number of observations"):
            a.assert_same_len(b)

    def test_slice_with_index_valid(self) -> None:
        X = np.arange(10)
        data = DummyData(X)

        ref = pd.MultiIndex.from_arrays(
            [list("abcdefghij")],
            names=["id"],
        )
        qry = pd.MultiIndex.from_arrays(
            [["b", "e", "j"]],
            names=["id"],
        )

        sliced = data.slice_with_index(ref, qry)

        np.testing.assert_array_equal(sliced.X, np.array([1, 4, 9]))

    def test_slice_with_index_returns_indices(self) -> None:
        X = np.arange(6)
        data = DummyData(X)

        ref = pd.MultiIndex.from_arrays([[0, 1, 2, 3, 4, 5]])
        qry = pd.MultiIndex.from_arrays([[2, 4]])

        sliced, idxs = data.slice_with_index(ref, qry, return_index=True)

        np.testing.assert_array_equal(idxs, np.array([2, 4]))
        np.testing.assert_array_equal(sliced.X, np.array([2, 4]))

    def test_slice_with_index_missing_key_raises(self) -> None:
        X = np.arange(5)
        data = DummyData(X)

        ref = pd.MultiIndex.from_arrays([[0, 1, 2, 3, 4]])
        qry = pd.MultiIndex.from_arrays([[2, 99]])

        with pytest.raises(KeyError, match="not present in reference index"):
            data.slice_with_index(ref, qry)

    def test_concat_collection_two_objects(self) -> None:
        """Concatenate two DummyData objects."""
        d1 = DummyData(np.random.randn(5, 3))
        d2 = DummyData(np.random.randn(7, 3))
        result = DummyData.concat_collection([d1, d2])
        assert len(result) == 12
        np.testing.assert_array_equal(result.X, np.vstack([d1.X, d2.X]))

    def test_concat_collection_three_objects(self) -> None:
        """Concatenate three DummyData objects."""
        d1 = DummyData(np.random.randn(2, 4))
        d2 = DummyData(np.random.randn(3, 4))
        d3 = DummyData(np.random.randn(4, 4))
        result = DummyData.concat_collection([d1, d2, d3])
        assert len(result) == 9
        expected = np.vstack([d1.X, d2.X, d3.X])
        np.testing.assert_array_equal(result.X, expected)

    def test_concat_collection_empty_raises(self) -> None:
        """Empty collection raises ValueError."""
        with pytest.raises(ValueError, match="Cannot concatenate empty collection"):
            DummyData.concat_collection([])

    def test_concat_collection_single_returns_new_instance(self) -> None:
        """Single-element concatenation returns a new instance with same data."""
        original = DummyData(np.random.randn(4, 2))
        result = DummyData.concat_collection([original])
        assert result is not original
        np.testing.assert_array_equal(result.X, original.X)

    def test_concat_collection_mismatched_dimensions_raises(self) -> None:
        """Mismatched number of columns raises ValueError."""
        d1 = DummyData(np.random.randn(5, 3))
        d2 = DummyData(np.random.randn(5, 4))  # different second dimension
        with pytest.raises(ValueError, match="Mismatched number of columns"):
            DummyData.concat_collection([d1, d2])

    def test_concat_collection_1d_arrays(self) -> None:
        """Handle 1D arrays (shape (n,))."""
        d1 = DummyData(np.random.randn(5))
        d2 = DummyData(np.random.randn(3))
        result = DummyData.concat_collection([d1, d2])
        assert len(result) == 8
        # For 1D, we treat columns=1 implicitly
        np.testing.assert_array_equal(result.X, np.concatenate([d1.X, d2.X]))
