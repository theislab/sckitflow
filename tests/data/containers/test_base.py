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
