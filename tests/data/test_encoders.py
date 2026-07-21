"""Unit + equivalence tests for the unified :mod:`sc_flow.core.data._encoders` abstraction (Change 2).

These pin the numerics to the pre-refactor path: a :class:`Lookup` reproduces the old ``.uns`` ``reps``
row-stack, and :class:`OneHot` reproduces scikit-learn's ``OneHotEncoder`` exactly.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.preprocessing import OneHotEncoder

from sc_flow.core.data._encoders import functional, label, lookup, one_hot


def test_lookup_reproduces_uns_row_stack():
    table = {"a": np.array([1.0, 2.0, 3.0]), "b": np.array([4.0, 5.0, 6.0])}
    enc = lookup("drug").fit(uns={"drug": table})
    out = enc.transform(np.array(["b", "a", "b"]))
    assert out.shape == (3, 3)
    np.testing.assert_array_equal(out[0], [4.0, 5.0, 6.0])
    np.testing.assert_array_equal(out[1], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(out[2], [4.0, 5.0, 6.0])


def test_lookup_needs_its_uns_table():
    with pytest.raises(KeyError):
        lookup("missing").fit(uns={})


def test_one_hot_matches_sklearn_exactly():
    vals = np.array(["x", "y", "x", "z"])
    enc = one_hot().fit(vals)
    ref = OneHotEncoder().fit(vals.reshape(-1, 1))
    query = np.array(["z", "x"])
    np.testing.assert_array_equal(enc.transform(query), ref.transform(query.reshape(-1, 1)).toarray())


def test_label_roundtrips():
    vals = np.array(["a", "b", "c", "a"])
    enc = label().fit(vals)
    codes = enc.transform(np.array(["c", "a"]))
    assert codes.shape == (2, 1)
    np.testing.assert_array_equal(enc.inverse_transform(codes), ["c", "a"])


def test_functional_applies_fn_and_inverse():
    enc = functional(fn=lambda x: x * 2.0, inv=lambda x: x / 2.0).fit(np.array([[1.0], [2.0]]))
    np.testing.assert_array_equal(enc.transform(np.array([[3.0]])), [[6.0]])
    np.testing.assert_array_equal(enc.inverse_transform(np.array([[6.0]])), [[3.0]])
