from dataclasses import dataclass
from typing import Any, ClassVar, TypeVar

import pandas as pd

from sc_flow.data._mixins import MappedLevelIndex, MappedTree
from sc_flow.data.containers._distribution import DistributionData

__all__ = ["MatchedData", "NestedData"]


T = TypeVar("T", bound=DistributionData)


@dataclass(frozen=True)
class MatchedData:
    """Container class for matched data."""

    target_distribution: T
    source_distribution: T | None = None

    def __post_init__(self) -> None:
        if self.source_distribution is not None:
            if (
                self.target_distribution.target_coupling_data is not None
                and self.source_distribution.source_coupling_data is not None
            ):
                self.target_distribution.target_coupling_data.assert_same_spatial_dims(
                    self.source_distribution.source_coupling_data
                )

    def __repr__(self) -> str:
        target_repr = "\n".join("\t" + line for line in repr(self.target_distribution).splitlines())
        parts = [f" * (target) -> {target_repr}"]

        if self.source_distribution is not None:
            source_repr = "\n".join("\t" + line for line in repr(self.source_distribution).splitlines())
            parts.append(f" * (source) -> {source_repr}")

        return f"{self.__class__.__name__}:\n" + "\n".join(parts)

    @property
    def target_distr(self) -> T:
        """Alias for :attr: `self.target_distribution`."""
        return self.target_distribution

    @property
    def source_distr(self) -> T | None:
        """Alias for :attr: `self.source_distribution`."""
        return self.source_distribution


@dataclass(frozen=True)
class NestedData(MappedTree):
    """Recursively mapped container for matched data."""

    required_key_type: ClassVar[type] = tuple
    required_value_type: ClassVar[type] = MatchedData

    @classmethod
    def init_from_data(
        cls,
        data: DistributionData,
        reference_index: pd.MultiIndex,
        mapped_index: MappedLevelIndex,
        source_key: tuple[Any] | None = None,
    ) -> "NestedData":
        """Initialized the recursive mapping from the input."""
        return cls._init_tree(data, reference_index, mapped_index, source_key)

    @classmethod
    def _init_leaf_node(
        cls,
        data: DistributionData,
        reference_index: pd.MultiIndex,
        mapped_index: MappedLevelIndex,
        source_key: tuple[Any] | None = None,
    ) -> "NestedData":
        if source_key is not None:
            source_idxs = mapped_index.mapping[source_key]
            source_distribution = data.slice_with_index(reference_index, source_idxs)
            rest_idxs = {k: v for k, v in mapped_index.mapping.items() if k != source_key}
        else:
            source_distribution = None
            rest_idxs = mapped_index.mapping
        return cls(
            {
                key: MatchedData(data.slice_with_index(reference_index, value), source_distribution=source_distribution)
                for key, value in rest_idxs.items()
            }
        )

    @classmethod
    def _init_tree(
        cls,
        data: DistributionData,
        reference_index: pd.MultiIndex,
        mapped_index: MappedLevelIndex,
        source_key: tuple[Any] | None = None,
    ) -> "NestedData":
        return cls(
            {
                key: cls._init_leaf_node(data, reference_index, value, source_key)
                if value.is_leaf
                else cls._init_tree(data, reference_index, value, source_key)
                for key, value in mapped_index.mapping.items()
            }
        )

    def flatten(self) -> list[MatchedData]:
        """Flattens itself into a list of :class: `MatchedData`"""
        if not self.is_leaf:
            return [v for val in self.mapping.values() for v in val.flatten()]
        return list(self.mapping.values())
