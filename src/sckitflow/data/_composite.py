from dataclasses import dataclass
from typing import Any, ClassVar, NamedTuple

from sckitflow._runtime import attempt_tqdm_import
from sckitflow.data._mixins import MappedLevelIndex, MappedTree
from sckitflow.data.containers._distribution import DistributionData

__all__ = ["MatchedData", "NestedData"]


class MatchedData(NamedTuple):
    """Container for matched data: a ``(target, source)`` distribution pair.

    ``source_distribution`` is ``None`` in the unpaired ("generate from noise")
    setting. For observation counts, use ``len(matched.target)`` /
    ``len(matched.source)`` directly.

    :param target_distribution: The target distribution for the matching.
    :type target_distribution: class: `DistributionData`

    :param source_distribution: Optional source distribution for the matching.
        Defaults to `None`.
    :type source_distribution: class: `DistributionData | None`
    """

    target_distribution: DistributionData
    source_distribution: DistributionData | None = None

    @property
    def target_distr(self) -> DistributionData:
        """Alias for :attr:`target_distribution`."""
        return self.target_distribution

    @property
    def source_distr(self) -> DistributionData | None:
        """Alias for :attr:`source_distribution`."""
        return self.source_distribution

    @property
    def target(self) -> DistributionData:
        """Alias for :attr:`target_distribution`."""
        return self.target_distribution

    @property
    def source(self) -> DistributionData | None:
        """Alias for :attr:`source_distribution`."""
        return self.source_distribution


@dataclass(frozen=True)
class NestedData(MappedTree):
    """Recursively mapped container for matched data."""

    _REQUIRED_KEY_TYPE: ClassVar[type] = tuple
    _REQUIRED_VALUE_TYPE: ClassVar[type] = MatchedData

    @classmethod
    def init_from_data(
        cls,
        data: DistributionData,
        mapped_index: MappedLevelIndex,
        source_key: tuple[Any] | None = None,
        matched_keys: dict[tuple[Any], tuple[Any]] | None = None,
    ) -> "NestedData":
        """Initialized the recursive mapping from the input.

        :param data: The flattened, unmatched distribution data.
        :type data: class: `DistributionData`

        :param mapped_index: The mapped tree of indices for each group.
        :type mapped_index: class: `MappedLevelIndex`

        :param source_key: Optional key used to identify the source groups for
            each leaf mapping. Defaults to `None`, in which case no source
            distribution will be considered.
        :type source_key: class: `tuple[Any] | None`

        :param matched_keys: Optional keys used to identify the source  and
            corresponding target groups in the case of fixed matches.
            When passed, takes precedence over :param: `source_key`.
            Defaults to `None`, in which case falls back to one to many coupling.
        :type matched_keys: class: `dict[tuple[Any], tuple[Any]] | None`
        """
        return cls._init_tree(data, mapped_index, source_key=source_key, matched_keys=matched_keys)

    @classmethod
    def _init_leaf_node_one_to_many(
        cls,
        data: DistributionData,
        mapped_index: MappedLevelIndex,
        source_key: tuple[Any] | None = None,
    ) -> "NestedData":
        """Initializes a leaf node given the input settings.

        Leaf values in mapped_index are slices into the sorted data,
        so indexing is O(1) per group with no search overhead.

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
        all_keys = list(mapped_index.mapping.keys())

        if source_key is not None:
            if source_key not in all_keys:
                raise KeyError(f"Source key {source_key} not ƒound.")
            source_distribution = data[mapped_index.mapping[source_key]]
            rest_keys = [k for k in all_keys if k != source_key]
        else:
            source_distribution = None
            rest_keys = all_keys

        tqdm = attempt_tqdm_import()
        pbar = tqdm(rest_keys) if tqdm is not None else None

        data_dict = {}
        for key in rest_keys:
            if pbar is not None:
                pbar.update()
            data_dict[key] = MatchedData(
                data[mapped_index.mapping[key]],
                source_distribution=source_distribution,
            )
        return cls(data_dict)

    @classmethod
    def _init_leaf_node_one_to_one(
        cls,
        data: DistributionData,
        mapped_index: MappedLevelIndex,
        matched_keys: dict[tuple[Any], tuple[Any]],
    ) -> "NestedData":
        # initialize progress bar
        tqdm = attempt_tqdm_import()
        pbar = tqdm(matched_keys) if tqdm is not None else None

        # iterate over the matched keys
        data_dict = {}
        for source_key, target_key in matched_keys.items():
            if pbar is not None:
                pbar.update()

            # check that keys are present
            if source_key not in mapped_index.mapping:
                raise KeyError(f"Source key {source_key} not ƒound.")
            if target_key not in mapped_index.mapping:
                raise KeyError(f"Target key {target_key} not ƒound.")

            # get distributions
            source_distribution = data[mapped_index.mapping[source_key]]
            target_distribution = data[mapped_index.mapping[target_key]]

            # initialize matched distribution
            matched_data = MatchedData(target_distribution, source_distribution=source_distribution)

            # store data
            key = (source_key, target_key)
            data_dict[key] = matched_data
        return cls(data_dict)

    @classmethod
    def _init_leaf_node(
        cls,
        data: DistributionData,
        mapped_index: MappedLevelIndex,
        source_key: tuple[Any] | None = None,
        matched_keys: dict[tuple[Any], tuple[Any]] | None = None,
    ) -> "NestedData":
        if matched_keys is not None:
            return cls._init_leaf_node_one_to_one(data, mapped_index, matched_keys=matched_keys)
        return cls._init_leaf_node_one_to_many(
            data,
            mapped_index,
            source_key=source_key,
        )

    @classmethod
    def _init_tree(
        cls,
        data: DistributionData,
        mapped_index: MappedLevelIndex,
        source_key: tuple[Any] | None = None,
        matched_keys: dict[tuple[Any], tuple[Any]] | None = None,
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
                key: cls._init_leaf_node(data, value, source_key=source_key, matched_keys=matched_keys)
                if value.is_leaf
                else cls._init_tree(data, value, source_key=source_key, matched_keys=matched_keys)
                for key, value in mapped_index.mapping.items()
            }
        )
