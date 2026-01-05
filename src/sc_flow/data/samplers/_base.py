import abc
from collections.abc import Callable
from functools import cached_property
from typing import TypeVar

import numpy as np

from sc_flow._constants import DEFAULT_BATCH_SIZE, DEFAULT_N_GROUPS
from sc_flow.data._composite import DistributionDType, MatchedData, NestedData
from sc_flow.data._utils import sample_indices_uniformly

__all__ = ["TreeDType", "BatchDType", "Sampler", "FSampler"]


TreeDType = TypeVar("CollectionDType", bound=NestedData)
BatchDType = TypeVar("BatchDType", bound=MatchedData)


class Sampler(abc.ABC):
    """Abstract base class for sampler objects.

    Subclasses need to override the :method _dispatch_sample: method.

    :param BATCH_DATA_CLS: The class used to store the matched distributions.
    :type BATCH_DATA_CLS: class: `type[BatchDType]`
    """

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

        :param tree: Tree storing the split and matched subpopulations.
        :type tree: class: `TreeDType`

        :param replace_samples: Whether to sample observations with replacement
            from each node. Defaults to `False`.
        :type replace_samples: class: `bool`

        :param replace_nodes: Whether to sample nodes with replacement
            from the tree. Defaults to `False`.
        :type replace_nodes: class: `bool`

        :param use_nodes_weights: Whether to use nodes weights in order to sample them.
            Each node weight is computed as its relative frequency over the whole tree.
            In order to compute the relative frequency of a node of matched distributions,
            only the target one is considered. Defaults to `True`.
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

    def _sample_indices_uniformly(
        self,
        n_obs: int,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> np.ndarray:
        """Samples a random mask to select a batch of observations from a distribution.

        :param batch_size: The number of observations to load in the batch.
        :type batch_size: class: `int`
        """
        return sample_indices_uniformly(n_obs, batch_size, replace=self._replace_samples)

    def _sample_from_distr(self, distr: DistributionDType, batch_size: int = DEFAULT_BATCH_SIZE) -> DistributionDType:
        mask = self._sample_indices_uniformly(len(distr), batch_size)
        return distr[mask]

    def _sample_observations(
        self,
        group: BatchDType,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> BatchDType:
        """Samples a batch of observations from a node of matched distributions."""
        # retrieve individual distributions
        target_distr: DistributionDType = group.target
        source_distr: DistributionDType | None = group.source

        # sample masks and slice
        target_distr = self._sample_from_distr(target_distr, batch_size)
        if source_distr is not None:
            source_distr = self._sample_from_distr(source_distr, batch_size)
        return self.BATCH_DATA_CLS(target_distr, source_distribution=source_distr)

    def _sample(
        self,
        n_nodes: int = DEFAULT_N_GROUPS,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> np.ndarray[BatchDType]:
        """Samples a batch of data from the tree.

        Sampling is done hierarchically, whereby a sed of nodes is sampled first
        and a subset of observations is selected from each sampled node.

        :param n_nodes: The number of nodes to sample.
        :type n_nodes: class: `int`

        :param batch_size: The number of observations to load in the batch.
        :type batch_size: class: `int`
        """
        groups = self._sample_nodes(n_nodes)
        return np.vectorize(self._sample_observations)(groups, batch_size)

    @property
    def tree(self) -> NestedData:
        """Returns the underlying data tree."""
        return self._tree

    @property
    def replace_samples(self) -> bool:
        """Exposes the :param replace_samples: attribute set at initialization."""
        return self._replace_samples

    @property
    def replace_nodes(self) -> bool:
        """Exposes the :param replace_nodes: attribute set at initialization."""
        return self._replace_nodes

    @property
    def use_nodes_weights(self) -> bool:
        """Exposes the :param use_nodes_weights: attribute set at initialization."""
        return self._use_nodes_weights

    @cached_property
    def flattened_data(self) -> list[BatchDType]:
        """Caches the flattened array of leaf nodes of the data tree."""
        return self._tree.flatten()

    @cached_property
    def groups_p(self) -> np.ndarray:
        """Returns the array of relative frequencies of each node.

        The relative frequency is computed by taking into account the number of observations
        in the target distribution.
        """

        def _get_len_target(e: BatchDType):
            return len(e.target)

        counts = np.vectorize(_get_len_target)(self.flattened_data)
        return counts / counts.sum()


class FSampler(Sampler):
    """Concrete class using an input callable to process the batch."""

    def __init__(
        self,
        data: TreeDType,
        f: Callable[[BatchDType], BatchDType],
        replace_samples: bool = False,
        replace_nodes: bool = False,
        use_nodes_weights: bool = True,
    ) -> None:
        """Initializes the sampler.

        :param tree: Tree storing the split and matched subpopulations.
        :type tree: class: `TreeDType`

        :param f: The function used to post-process the batch of matched data.
        :type f: class: `Callable[[BatchDType], BatchDType]`

        :param replace_samples: Whether to sample observations with replacement
            from each node. Defaults to `False`.
        :type replace_samples: class: `bool`

        :param replace_nodes: Whether to sample nodes with replacement
            from the tree. Defaults to `False`.
        :type replace_nodes: class: `bool`

        :param use_nodes_weights: Whether to use nodes weights in order to sample them.
            Each node weight is computed as its relative frequency over the whole tree.
            In order to compute the relative frequency of a node of matched distributions,
            only the target one is considered. Defaults to `True`.
        :type use_nodes_weights: class: `bool`
        """
        super().__init__(
            data, replace_samples=replace_samples, replace_nodes=replace_nodes, use_nodes_weights=use_nodes_weights
        )
        self._f = f

    def _dispatch_sample(self, group):
        """Processes the batch before returning it."""
        return self._f(group)

    @property
    def f(self) -> Callable[[BatchDType], BatchDType]:
        """Exposes the :param f: attribute set at initialization."""
        return self._f
