from __future__ import annotations

import abc
from collections.abc import Iterable
from typing import Generic, TypeVar

__all__ = [
    "Distribution",
    "MatchedDistributions",
    "DataTree",
    "DataT",
    "DistributionT",
    "MatchedDistributionsT",
    "DataTreeT",
]


DataT = TypeVar("DataT")
DistributionT = TypeVar("DistributionDType", bound="Distribution")
MatchedDistributionsT = TypeVar("MatchedDistributionsT", bound="MatchedDistributions")
DataTreeT = TypeVar("DataTreeT", bound="DataTree")


class Distribution(abc.ABC):
    @abc.abstractmethod
    def __len__(self) -> int: ...

    @abc.abstractmethod
    def __getitem__(self, idx) -> Distribution: ...


class MatchedDistributions(abc.ABC):
    target: Distribution
    source: Distribution | None


class DataTree(Generic[DataT], abc.ABC):
    @abc.abstractmethod
    def flatten(self) -> Iterable[MatchedDistributions]: ...
