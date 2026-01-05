from dataclasses import dataclass
from typing import Any, ClassVar, TypeVar

import pandas as pd

from sc_flow._runtime import attempt_tqdm_import
from sc_flow.data._mixins import MappedLevelIndex, MappedTree
from sc_flow.data.containers._distribution import DistributionData

__all__ = ["DistributionDType", "MatchedData", "NestedData"]


DistributionDType = TypeVar("T", bound=DistributionData)


@dataclass(frozen=True)
class MatchedData:
    """Container class for matched data."""

    target_distribution: DistributionDType
    source_distribution: DistributionDType | None = None

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
    def target_distr(self) -> DistributionDType:
        """Alias for :attr: `self.target_distribution`."""
        return self.target_distribution

    @property
    def source_distr(self) -> DistributionDType | None:
        """Alias for :attr: `self.source_distribution`."""
        return self.source_distribution

    @property
    def target(self) -> DistributionDType:
        """Alias for :attr: `self.target_distribution`."""
        return self.target_distribution

    @property
    def source(self) -> DistributionDType | None:
        """Alias for :attr: `self.source_distribution`."""
        return self.source_distribution

    @property
    def n_source_obs(self) -> int | None:
        """"""  # noqa
        if self.source_distribution is None:
            return None
        return self.source_distribution.n_obs

    @property
    def n_target_obs(self) -> int:
        """"""  # noqa
        return self.target_distribution.n_obs

    @property
    def n_src_obs(self) -> int:
        """"""  # noqa
        return self.n_source_obs

    @property
    def n_tgt_obs(self) -> int:
        """"""  # noqa
        return self.n_target_obs


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
        # split source distribution apart
        if source_key is not None:
            source_idxs = mapped_index.mapping[source_key]
            source_distribution = data.slice_with_index(reference_index, source_idxs)
            rest_idxs = {k: v for k, v in mapped_index.mapping.items() if k != source_key}
        else:
            source_distribution = None
            rest_idxs = mapped_index.mapping

        # lazily import tqdm
        tqdm = attempt_tqdm_import()
        if tqdm is not None:
            pbar = tqdm(rest_idxs)
        else:
            pbar = None

        # construct data dictionary
        data_dict = {}
        for key, value in rest_idxs.items():
            # update progress bar
            if pbar is not None:
                pbar.update()

            # update dictionary
            data_dict[key] = MatchedData(
                data.slice_with_index(reference_index, value), source_distribution=source_distribution
            )
        return cls(data_dict)

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
