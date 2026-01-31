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
    """Container class for matched data.

    :param target_distribution: The target distribution for the matching.
    :type target_distribution: class: `DistributionDType`

    :param source_distribution: Optional source distribution for the matching.
        Defaults to `None`.
    :type source_distribution: class: `DistributionDType | None`
    """

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
        """Returns the number of observations in the source distribution.

        When the source distribution is not provided, it will return `None`.
        """
        if self.source_distribution is None:
            return None
        return len(self.source_distribution)

    @property
    def n_target_obs(self) -> int:
        """Returns the number of observations in the target distribution."""
        return len(self.target_distribution)

    @property
    def n_src_obs(self) -> int:
        """Alias for :attr: `self.n_source_obs`."""
        return self.n_source_obs

    @property
    def n_tgt_obs(self) -> int:
        """Alias for :attr: `self.n_target_obs`."""
        return self.n_target_obs


@dataclass(frozen=True)
class NestedData(MappedTree):
    """Recursively mapped container for matched data."""

    _REQUIRED_KEY_TYPE: ClassVar[type] = tuple
    _REQUIRED_VALUE_TYPE: ClassVar[type] = MatchedData

    @classmethod
    def init_from_data(
        cls,
        data: DistributionData,
        reference_index: pd.MultiIndex,
        mapped_index: MappedLevelIndex,
        source_key: tuple[Any] | None = None,
    ) -> "NestedData":
        """Initialized the recursive mapping from the input.

        :param data: The flattened, unmatched distribution data.
        :type data: class: `DistributionData`

        :param reference_index: The reference index of the unmatched data,
            needed to split it into groups.
        :type reference_index: class: `pd.MultiIndex`

        :param mapped_index: The mapped tree of indices for each group.
        :type mapped_index: class: `MappedLevelIndex`

        :param source_key: Optional key used to identify the source groups for
            each leaf mapping. Defaults to `None`, in which case no source
            distribution will be considered.
        :type source_key: class: `tuple[Any] | None`
        """
        return cls._init_tree(data, reference_index, mapped_index, source_key)

    @classmethod
    def _init_leaf_node(
        cls,
        data: DistributionData,
        reference_index: pd.MultiIndex,
        mapped_index: MappedLevelIndex,
        source_key: tuple[Any] | None = None,
    ) -> "NestedData":
        """Initializes a leaf node given the input settings.

        :param data: The flattened, unmatched distribution data.
        :type data: class: `DistributionData`

        :param reference_index: The reference index of the unmatched data,
            needed to split it into groups.
        :type reference_index: class: `pd.MultiIndex`

        :param mapped_index: The mapped tree of indices for each group.
        :type mapped_index: class: `MappedLevelIndex`

        :param source_key: Optional key used to identify the source groups for
            each leaf mapping. Defaults to `None`, in which case no source
            distribution will be considered.
        :type source_key: class: `tuple[Any] | None`
        """
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
        """Initializes the tree from the given settings.

        :param data: The flattened, unmatched distribution data.
        :type data: class: `DistributionData`

        :param reference_index: The reference index of the unmatched data,
            needed to split it into groups.
        :type reference_index: class: `pd.MultiIndex`

        :param mapped_index: The mapped tree of indices for each group.
        :type mapped_index: class: `MappedLevelIndex`

        :param source_key: Optional key used to identify the source groups for
            each leaf mapping. Defaults to `None`, in which case no source
            distribution will be considered.
        :type source_key: class: `tuple[Any] | None`
        """
        return cls(
            {
                key: cls._init_leaf_node(data, reference_index, value, source_key)
                if value.is_leaf
                else cls._init_tree(data, reference_index, value, source_key)
                for key, value in mapped_index.mapping.items()
            }
        )
