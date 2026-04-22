from dataclasses import asdict, dataclass
from typing import Any, ClassVar

from sc_flow._runtime import attempt_tqdm_import
from sc_flow.data._abc import DistributionT, MatchedDistributions
from sc_flow.data._mixins import MappedLevelIndex, MappedTree
from sc_flow.data.containers._distribution import DistributionData

__all__ = ["DistributionT", "MatchedData", "NestedData"]


@dataclass(frozen=True)
class MatchedData(MatchedDistributions):
    """Container class for matched data.

    :param target_distribution: The target distribution for the matching.
    :type target_distribution: class: `DistributionT`

    :param source_distribution: Optional source distribution for the matching.
        Defaults to `None`.
    :type source_distribution: class: `DistributionT | None`
    """

    target_distribution: DistributionT
    source_distribution: DistributionT | None = None

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

    def to_dict(self) -> dict[str, Any]:
        """"""  # noqa
        return asdict(self)

    @property
    def target_distr(self) -> DistributionT:
        """Alias for :attr: `self.target_distribution`."""
        return self.target_distribution

    @property
    def source_distr(self) -> DistributionT | None:
        """Alias for :attr: `self.source_distribution`."""
        return self.source_distribution

    @property
    def target(self) -> DistributionT:
        """Alias for :attr: `self.target_distribution`."""
        return self.target_distribution

    @property
    def source(self) -> DistributionT | None:
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
        mapped_index: MappedLevelIndex,
        source_key: tuple[Any] | None = None,
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
        """
        return cls._init_tree(data, mapped_index, source_key)

    @classmethod
    def _init_leaf_node(
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
    def _init_tree(
        cls,
        data: DistributionData,
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
                key: cls._init_leaf_node(data, value, source_key)
                if value.is_leaf
                else cls._init_tree(data, value, source_key)
                for key, value in mapped_index.mapping.items()
            }
        )
