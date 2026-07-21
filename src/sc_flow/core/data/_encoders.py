"""One encoder abstraction (schema-generalization Change 2).

`reps` (a ``.uns`` lookup table) and `encoding` (a fit-from-data transform like ``one-hot``) are the
same idea — turn a categorical column's values into per-value vectors and back — differing only in
*where the transform's parameters come from*. This module collapses the old four parallel dicts
(``_reps`` / ``_encoding`` / ``_encoding_transform_fn`` / ``_encoding_inverse_transform_fn``) into a
single map ``{col: Encoder}`` shared by conditions, covariates, and response.

An :class:`Encoder` is fittable and exposes the two directions ``transform`` / ``inverse_transform``.
``fit`` receives the column's values (for data-fit encoders) and the ``.uns`` mapping (for a
:class:`Lookup`, whose parameters live there). Factories: :func:`one_hot`, :func:`label`,
:func:`lookup`, :func:`functional`.

The numerics are deliberately identical to the pre-refactor path: :class:`OneHot`/:class:`Label`/
:class:`Functional` wrap the same scikit-learn estimators built in :mod:`sc_flow.core.data._utils`, and
:class:`Lookup` reproduces ``CategoricalData._extract_col_repr`` (per-value ``.uns`` row, stacked).
"""

from __future__ import annotations

import abc
from collections.abc import Callable, Hashable, Mapping

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, OneHotEncoder

__all__ = ["Encoder", "OneHot", "Label", "Lookup", "Functional", "one_hot", "label", "lookup", "functional"]


def _as_1d(values: np.ndarray) -> np.ndarray:
    return np.asarray(values).reshape(-1)


class Encoder(abc.ABC):
    """A fittable column encoder with two directions.

    ``transform`` maps a 1-D array of category values → ``(n, dim)``; ``inverse_transform`` maps back.
    ``fit`` is given the column ``values`` (data-fit encoders) and/or the ``.uns`` mapping (lookups).
    """

    @abc.abstractmethod
    def fit(self, values: np.ndarray | None = None, *, uns: Mapping[str, object] | None = None) -> Encoder:
        """Fit the encoder from column ``values`` and/or the ``.uns`` mapping; returns ``self``."""

    @abc.abstractmethod
    def transform(self, values: np.ndarray) -> np.ndarray:
        """Encode a 1-D array of category values into a ``(n, dim)`` float array."""

    @abc.abstractmethod
    def inverse_transform(self, arr: np.ndarray) -> np.ndarray:
        """Decode a ``(n, dim)`` array back to the original category values."""


class _SklearnEncoder(Encoder):
    """Base for encoders backed by a scikit-learn estimator fit on the column values."""

    def __init__(self) -> None:
        self._est = None

    def transform(self, values: np.ndarray) -> np.ndarray:
        out = self._est.transform(self._shape_in(values))
        if isinstance(out, csr_matrix):
            out = out.toarray()
        return np.asarray(out)

    def inverse_transform(self, arr: np.ndarray) -> np.ndarray:
        return self._est.inverse_transform(arr)

    def _shape_in(self, values: np.ndarray) -> np.ndarray:  # pragma: no cover - overridden
        raise NotImplementedError


class OneHot(_SklearnEncoder):
    """One-hot encoding fit on the column's observed categories (scikit-learn ``OneHotEncoder``)."""

    def fit(self, values: np.ndarray | None = None, *, uns: Mapping[str, object] | None = None) -> OneHot:
        data = np.asarray(values).reshape(-1, 1)
        self._est = OneHotEncoder().fit(data)
        return self

    def _shape_in(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values).reshape(-1, 1)


class Label(_SklearnEncoder):
    """Integer label encoding (scikit-learn ``LabelEncoder``); yields a ``(n, 1)`` column."""

    def fit(self, values: np.ndarray | None = None, *, uns: Mapping[str, object] | None = None) -> Label:
        self._est = LabelEncoder().fit(_as_1d(values))
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        return self._est.transform(_as_1d(values)).reshape(-1, 1)

    def inverse_transform(self, arr: np.ndarray) -> np.ndarray:
        return self._est.inverse_transform(_as_1d(arr))


class Functional(_SklearnEncoder):
    """A user-supplied transform / inverse pair (scikit-learn ``FunctionTransformer``)."""

    def __init__(self, fn: Callable | None = None, inv: Callable | None = None) -> None:
        super().__init__()
        self._fn = fn
        self._inv = inv

    def fit(self, values: np.ndarray | None = None, *, uns: Mapping[str, object] | None = None) -> Functional:
        self._est = FunctionTransformer(func=self._fn, inverse_func=self._inv, check_inverse=False).fit(values)
        return self

    def _shape_in(self, values: np.ndarray) -> np.ndarray:
        return values


class Lookup(Encoder):
    """A ``.uns`` lookup table encoder — an embedding whose parameters live in ``adata.uns[uns_key]``.

    Reproduces the old ``reps`` path: each value maps to its row ``uns[uns_key][value]`` (reshaped to a
    row vector), stacked over the input. ``inverse_transform`` nearest-matches against the table rows.
    """

    def __init__(self, uns_key: str) -> None:
        self.uns_key = uns_key
        self._table: Mapping[Hashable, np.ndarray] | None = None

    def fit(self, values: np.ndarray | None = None, *, uns: Mapping[str, object] | None = None) -> Lookup:
        if uns is None or self.uns_key not in uns:
            raise KeyError(f"Lookup encoder needs adata.uns[{self.uns_key!r}] to fit.")
        self._table = uns[self.uns_key]
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self._table is None:
            raise RuntimeError("Lookup encoder used before fit(uns=...).")
        rows = [np.asarray(self._table[v]).reshape(1, -1) for v in _as_1d(values)]
        return np.vstack(rows)

    def inverse_transform(self, arr: np.ndarray) -> np.ndarray:
        if self._table is None:
            raise RuntimeError("Lookup encoder used before fit(uns=...).")
        keys = list(self._table)
        table = np.vstack([np.asarray(self._table[k]).reshape(1, -1) for k in keys])
        arr = np.atleast_2d(arr)
        idx = np.array([int(np.argmin(np.linalg.norm(table - row, axis=1))) for row in arr])
        return np.array([keys[i] for i in idx], dtype=object)


def one_hot() -> OneHot:
    """One-hot encoder, fit on the column's observed categories."""
    return OneHot()


def label() -> Label:
    """Integer label encoder."""
    return Label()


def lookup(uns_key: str) -> Lookup:
    """Encoder that reads per-value vectors from ``adata.uns[uns_key]`` (the old ``reps`` path)."""
    return Lookup(uns_key)


def functional(fn: Callable | None = None, inv: Callable | None = None) -> Functional:
    """Encoder wrapping a user ``fn`` / inverse ``inv`` pair."""
    return Functional(fn=fn, inv=inv)
