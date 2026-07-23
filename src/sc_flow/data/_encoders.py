
from __future__ import annotations

import abc
from collections.abc import Callable, Hashable, Mapping

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, OneHotEncoder

__all__ = [
    "Encoder", "Categorical", "OneHot", "Label", "Lookup", "Functional",
    "categorical", "one_hot", "label", "lookup", "functional",
    "build_encoder",
]


def _as_1d(values: np.ndarray) -> np.ndarray:
    return np.asarray(values).reshape(-1)


class Encoder(abc.ABC):

    @abc.abstractmethod
    def fit(self, values: np.ndarray | None = None, *, uns: Mapping[str, object] | None = None) -> Encoder:
        ...

    @abc.abstractmethod
    def transform(self, values: np.ndarray) -> np.ndarray:
        ...

    @abc.abstractmethod
    def inverse_transform(self, arr: np.ndarray) -> np.ndarray:
        ...


class _SklearnEncoder(Encoder):

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

    def fit(self, values: np.ndarray | None = None, *, uns: Mapping[str, object] | None = None) -> OneHot:
        data = np.asarray(values).reshape(-1, 1)
        self._est = OneHotEncoder().fit(data)
        return self

    def _shape_in(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values).reshape(-1, 1)


class Label(_SklearnEncoder):

    def fit(self, values: np.ndarray | None = None, *, uns: Mapping[str, object] | None = None) -> Label:
        self._est = LabelEncoder().fit(_as_1d(values))
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        return self._est.transform(_as_1d(values)).reshape(-1, 1)

    def inverse_transform(self, arr: np.ndarray) -> np.ndarray:
        return self._est.inverse_transform(_as_1d(arr))


class Categorical(Label):
    """Canonical data-side declaration that a realm is **categorical**: the dataloader emits an integer
    index and the *model* owns the encoding (learned embedding or one-hot — chosen in the model config).
    ``one_hot``/``label`` remain as aliases meaning the same thing on the data side now.
    """


class Functional(_SklearnEncoder):

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


def categorical() -> Categorical:
    """A categorical realm → the model receives an integer index (embeds/one-hots it). Canonical."""
    return Categorical()


def one_hot() -> OneHot:
    return OneHot()


def label() -> Label:
    return Label()


def lookup(uns_key: str) -> Lookup:
    return Lookup(uns_key)


def functional(fn: Callable | None = None, inv: Callable | None = None) -> Functional:
    return Functional(fn=fn, inv=inv)


def build_encoder(spec: str) -> Encoder:
    """Resolve a string data-side encoder ``spec`` to an :class:`Encoder` (the config-driven counterpart
    of the model-side ``build_realm_encoder``, so a pipeline author never hand-writes ``_make_encoder``).

    Recognized: ``"categorical"`` / ``"one_hot"`` / ``"label"`` (parameter-free) and ``"lookup:<uns_key>"``
    (the looked-up feature table in ``adata.uns[<uns_key>]``).
    """
    if spec == "categorical":
        return categorical()
    if spec == "one_hot":
        return one_hot()
    if spec == "label":
        return label()
    if spec.startswith("lookup:"):
        return lookup(spec.split(":", 1)[1])
    raise ValueError(
        f"unknown encoder spec {spec!r}; expected 'categorical', 'one_hot', 'label', or 'lookup:<uns_key>'."
    )
