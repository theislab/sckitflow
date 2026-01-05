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
        replace_nodes: bool = False,
        use_nodes_weights: bool = True,
    ) -> None:
        """Initializes the sampler.

        :param tree:
        :type tree: class: `TreeDType`

        :param replace_samples:
        :type replace_samples: class: `bool`

        :param replace_nodes:
        :type replace_nodes: class: `bool`

        :param use_nodes_weights:
        :type use_nodes_weights: class: `bool`
        """
        self._tree = tree
        self._replace_samples = replace_samples
        self._replace_nodes = replace_nodes
        self._use_nodes_weights = use_nodes_weights

    @abc.abstractmethod
    def _dispatch_sample(self, group: BatchDType) -> BatchDType:
        """Processes the batch before returning it."""

    def _sample_nodes(
        self,
        n_nodes: int,
    ) -> np.ndarray[BatchDType]:
        """Samples an array of leaf nodes from the tree.

        :param n_nodes: The number of nodes to sample.
        :type n_nodes: class: `int`
        """
        return np.random.choice(
            self.flattened_data,
            n_nodes,
            p=self.groups_p if self._use_nodes_weights else None,
            replace=self._replace_nodes,
        )

    # possible bottleneck
    def _sample_mask(
        self,
        distr: DistributionDType,
        batch_size: int,
    ) -> np.ndarray:
        """Samples a random mask to select a batch of observations from a node.

        :param batch_size:
        """
        return np.random.choice(len(distr), batch_size, replace=self._replace_samples)

    def _sample_observations(
        self,
        group: BatchDType,
        batch_size: int,
    ) -> BatchDType:
        """"""  # noqa
        # retrieve individual distributions
        target_distr: DistributionDType = group.target
        source_distr: DistributionDType | None = group.source

        # sample masks and slice
        target_mask = self._sample_mask(target_distr, batch_size)
        target_distr = target_distr[target_mask]
        if source_distr is not None:
            source_mask = self._sample_mask(source_distr, batch_size)
            source_distr = source_distr[source_mask]
        return self.BATCH_DATA_CLS(target_distr, source_distribution=source_distr)

    def _sample(
        self,
        n_nodes: int,
        batch_size: int,
    ) -> np.ndarray[BatchDType]:
        """"""  # noqa
        groups = self._sample_nodes(n_nodes)
        return np.vectorize(self._sample_observations)(groups, batch_size)

    @property
    def tree(self) -> NestedData:
        """"""  # noqa
        return self._tree

    @property
    def replace_samples(self) -> bool:
        return self._replace_samples

    @property
    def replace_nodes(self) -> bool:
        return self._replace_nodes

    @property
    def use_nodes_weights(self) -> bool:
        return self._use_nodes_weights

    @cached_property
    def flattened_data(self) -> list[BatchDType]:
        """"""  # noqa
        return self._tree.flatten()

    @cached_property
    def groups_p(self) -> np.ndarray:
        """"""  # noqa

        def _get_len_target(e: BatchDType):
            return len(e.target)

        counts = np.vectorize(_get_len_target)(self.flattened_data)
        return counts / counts.sum()


class FSampler(Sampler):
    """"""  # noqa

    def __init__(
        self,
        data: TreeDType,
        f: Callable[[BatchDType], BatchDType],
        replace_samples: bool = False,
        replace_nodes: bool = False,
        use_nodes_weights: bool = True,
    ) -> None:
        """"""  # noqa
        super().__init__(
            data, replace_samples=replace_samples, replace_nodes=replace_nodes, use_nodes_weights=use_nodes_weights
        )
        self._f = f

    def _dispatch_sample(self, group):
        """"""  # noqa
        return self._f(group)

    @property
    def f(self) -> Callable[[BatchDType], BatchDType]:
        """"""  # noqa
        return self._f
