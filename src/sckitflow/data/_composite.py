from dataclasses import asdict, dataclass
from typing import Any, ClassVar

from sckitflow._runtime import attempt_tqdm_import
from sckitflow.data._abc import DistributionT, MatchedDistributions
from sckitflow.data._mixins import MappedLevelIndex, MappedTree
from sckitflow.data.containers._distribution import DistributionData

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

    def align(self) -> "MatchedData":
        """Aligns source and target distributions.

        As source and target distributions are not constrained to contain the same number of samples,
        this method is needed for such cases where such alignment is required. In particular,
        this is the case whenever both continuous conditioning covariates and source/control states are
        provided, as broadcasting would fail in this scenario.

        The logic for the alignment is the following:

        * No-op fallback for the cases in which no continuous condition covariate is provided,
            no source/control state is modeled (i.e.: generation from noise) or
            source and target distributions already have the same number of observations.
        * When the target distribution contains less observations than the source one, source states
            are simply sliced (subsetted) to the same amount of samples in the target. The subsetting
            is done following the order given by the array storage, hence one would need to
            sort the source/control states according to their needs in case they want to consider
            some specific control states.
        * When the target distribution contains more observations than the source one, source states
            are reapeated as many times as needed to match the number of target observations.
        """
        # return self when no continuous covariates are present
        if not self.target.has_continuous_condition_covariates:
            return self
        # return self when no source states are present
        if self.source is None:
            return self

        # get number of observation in source and target data
        n_src_obs = self.n_source_obs
        n_tgt_obs = self.n_target_obs

        # return self when they have same number of observations
        if n_tgt_obs == n_src_obs:
            return self

        # slice source when there are less observations
        if n_tgt_obs < n_src_obs:
            tgt_slice = slice(n_tgt_obs)
            src_data = self.source[tgt_slice]
            return MatchedData(
                self.target,
                source_distribution=src_data,
            )

        # repeat source as many times as needed
        n_repeats = n_tgt_obs // n_src_obs
        n_remainders = n_tgt_obs % n_src_obs

        # prepare for concatenation
        src_to_concat = []
        for _ in range(n_repeats):
            src_data = self.source
            src_to_concat.append(src_data)

        # append remainder
        src_data = self.source[slice(n_remainders)]
        src_to_concat.append(src_data)

        # concatenate distribution data
        src_data = DistributionData.concat_collection(src_to_concat)
        return MatchedData(
            self.target,
            source_distribution=src_data,
        )

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
