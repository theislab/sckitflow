import abc
from collections.abc import Callable
from functools import cached_property
from typing import TypeVar

import numpy as np

from sc_flow.data._composite import DistributionDType, MatchedData, NestedData

__all__ = ["TreeDType", "BatchDType", "Sampler", "FSampler"]


TreeDType = TypeVar("CollectionDType", bound=NestedData)
BatchDType = TypeVar("BatchDType", bound=MatchedData)


class Sampler(abc.ABC):
    """Base class for sampler objects."""

    BATCH_DATA_CLS: type[BatchDType] = MatchedData

    def __init__(
        self,
        tree: TreeDType,
        *args,
        replace_samples: bool = False,
        replace_groups: bool = False,
        use_groups_weights: bool = True,
    ) -> None:
        """"""  # noqa
        self._tree = tree
        self._replace_samples = replace_samples
        self._replace_groups = replace_groups
        self._use_groups_weights = use_groups_weights

    @abc.abstractmethod
    def _dispatch_sample(self, group: BatchDType) -> BatchDType:
        """"""  # noqa

    def _sample_groups(
        self,
        n_groups: int,
    ) -> np.ndarray[BatchDType]:
        """"""  # noqa
        return np.random.choice(
            self.flattened_data,
            n_groups,
            p=self.groups_p if self._use_groups_weights else None,
            replace=self._replace_groups,
        )

    # possible bottleneck
    def _sample_mask(
        self,
        distr: DistributionDType,
        batch_size: int,
    ) -> np.ndarray:
        """"""  # noqa
        return np.random.choice(distr.n_obs, batch_size, replace=self._replace_samples)

    def _sample_observations(
        self,
        group: BatchDType,
        batch_size: int,
    ) -> BatchDType:
        """"""  # noqa
        # retrieve individual distributions
        target_distr: DistributionDType = group.target_distr
        source_distr: DistributionDType | None = group.source_distr

        # sample masks and slice
        target_mask = self._sample_mask(target_distr, batch_size)
        target_distr = target_distr.slice_with_array(target_mask)
        if source_distr is not None:
            source_mask = self._sample_mask(source_distr, batch_size)
            source_distr = source_distr.slice_with_array(source_mask)
        return self.BATCH_DATA_CLS(target_distr, source_distribution=source_distr)

    def _sample(
        self,
        n_groups: int,
        batch_size: int,
    ) -> np.ndarray[BatchDType]:
        """"""  # noqa
        groups = self._sample_groups(n_groups)
        return np.vectorize(self._sample_observations)(groups, batch_size)

    @property
    def tree(self) -> NestedData:
        """"""  # noqa
        return self._tree

    @property
    def replace_samples(self) -> bool:
        return self._replace_samples

    @property
    def replace_groups(self) -> bool:
        return self._replace_groups

    @property
    def use_groups_weights(self) -> bool:
        return self._use_groups_weights

    @cached_property
    def flattened_data(self) -> list[BatchDType]:
        """"""  # noqa
        return self._tree.flatten()

    @cached_property
    def groups_p(self) -> np.ndarray:
        """"""  # noqa
        counts = np.vectorize(lambda e: e.n_target_obs)(self.flattened_data)
        return counts / counts.sum()


class FSampler(Sampler):
    """"""  # noqa

    def __init__(
        self,
        data: TreeDType,
        f: Callable[[BatchDType], BatchDType],
        replace_samples: bool = False,
        replace_groups: bool = False,
        use_groups_weights: bool = True,
    ) -> None:
        """"""  # noqa
        super().__init__(
            data, replace_samples=replace_samples, replace_groups=replace_groups, use_groups_weights=use_groups_weights
        )
        self._f = f

    def _dispatch_sample(self, group):
        """"""  # noqa
        return self._f(group)

    @property
    def f(self) -> Callable[[BatchDType], BatchDType]:
        """"""  # noqa
        return self._f
