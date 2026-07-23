from __future__ import annotations

import abc
from collections.abc import Iterable
from typing import Generic, TypeVar

__all__ = [
    "DataTree",
    "DataT",
    "DataTreeT",
]


DataT = TypeVar("DataT")
DataTreeT = TypeVar("DataTreeT", bound="DataTree")


class DataTree(Generic[DataT], abc.ABC):
    @abc.abstractmethod
    def flatten(self) -> Iterable[DataT]: ...
