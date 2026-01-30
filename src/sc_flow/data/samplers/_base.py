import abc
from collections.abc import Callable
from functools import cached_property, partial
from typing import Generic, TypeVar

import numpy as np

from sc_flow._constants import DEFAULT_BATCH_SIZE, DEFAULT_N_GROUPS
from sc_flow.data._composite import DistributionDType, MatchedData, NestedData

__all__ = ["TreeDType", "NodeDType", "BatchDType", "Sampler", "FSampler"]


TreeDType = TypeVar("CollectionDType", bound=NestedData)
NodeDType = TypeVar("NodeDType", bound=MatchedData)
BatchDType = TypeVar("BatchDType")


class Sampler(Generic[NodeDType, BatchDType], abc.ABC):
    """Abstract base class for sampler objects on trees.

    Subclasses need to override the :method _dispatch_sample: method.
    """

    def __init__(
        self,
        tree: TreeDType,
        *args,
        replace_samples: bool = False,
        replace_nodes: bool = False,
        use_nodes_weights: bool = True,
        **kwargs,
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
    def _preprocess_sample(self, group: NodeDType) -> BatchDType:
        """Processes all the batches before sampling"""

    @abc.abstractmethod
    def _dispatch_sample(self, group: NodeDType) -> BatchDType:
        """Processes the batch before returning it."""

    def _sample_nodes(
        self,
        n_nodes: int,
    ) -> np.ndarray[NodeDType]:
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

    def _sample_indices(
        self,
        n_obs: int,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> np.ndarray:
        """Samples a random mask to select a batch of observations from a distribution.

        :param batch_size: The number of observations to load in the batch.
        :type batch_size: class: `int`
        """
        if self._replace_samples:
            return np.random.randint(0, n_obs, batch_size)
        else:
            if batch_size > n_obs:
                msg = "Cannot take a larger sample than population."
                raise ValueError(msg)
            return np.random.permutation(n_obs)[:batch_size]

    def _sample_from_distr(self, distr: DistributionDType, batch_size: int = DEFAULT_BATCH_SIZE) -> DistributionDType:
        mask = self._sample_indices(len(distr), batch_size)
        return distr[mask]

    def _sample_observations(
        self,
        group: NodeDType,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> BatchDType:
        """Samples a batch of observations from a node of matched distributions."""
        target_distr = self._sample_from_distr(group.target, batch_size)
        if group.source is not None:
            source_distr = self._sample_from_distr(group.source, batch_size)
        else:
            source_distr = None
        batch_data = group.__class__(target_distr, source_distribution=source_distr)
        return self._dispatch_sample(batch_data)

    def _sample(
        self,
        n_nodes: int = DEFAULT_N_GROUPS,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> tuple[BatchDType]:
        """Samples a batch of data from the tree.

        Sampling is done hierarchically, whereby a sed of nodes is sampled first
        and a subset of observations is selected from each sampled node.

        :param n_nodes: The number of nodes to sample.
        :type n_nodes: class: `int`

        :param batch_size: The number of observations to load in the batch.
        :type batch_size: class: `int`
        """
        nodes = self._sample_nodes(n_nodes)
        sample_fn = partial(self._sample_observations, batch_size=batch_size)
        return tuple(map(sample_fn, nodes))

    @property
    def tree(self) -> TreeDType:
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
    def flattened_data(self) -> np.ndarray[NodeDType]:
        """Caches the flattened array of leaf nodes of the data tree."""
        return np.array([self._preprocess_sample(node) for node in self._tree.flatten()])

    @cached_property
    def groups_p(self) -> np.ndarray:
        """Returns the array of relative frequencies of each node.

        The relative frequency is computed by taking into account the number of observations
        in the target distribution.
        """
        counts = np.array([len(e.target) for e in self.flattened_data])
        return counts / counts.sum()


class FSampler(Sampler[MatchedData, BatchDType]):
    """Concrete class using an input callable to process the batch."""

    def __init__(
        self,
        data: TreeDType,
        dispatch_fn: Callable[[BatchDType], BatchDType],
        preprocess_fn: Callable[[NodeDType], NodeDType] | None = None,
        replace_samples: bool = False,
        replace_nodes: bool = False,
        use_nodes_weights: bool = True,
    ) -> None:
        """Initializes the sampler.

        :param tree: Tree storing the split and matched subpopulations.
        :type tree: class: `TreeDType`

        :param f: The function used to post-process the batch of matched data.
        :type f: class: `Callable[[NodeDType], NodeDType]`

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
        self._preprocess_fn = preprocess_fn if preprocess_fn is not None else lambda x: x
        self._dispatch_fn = dispatch_fn

    def _preprocess_sample(self, group: NodeDType) -> NodeDType:
        """Processes all the nodes before sampling from them."""
        return self._preprocess_fn(group)

    def _dispatch_sample(self, group: NodeDType) -> BatchDType:
        """Processes the batch before returning it."""
        return self._dispatch_fn(group)

    @property
    def preprocess_fn(self) -> Callable[[NodeDType], BatchDType]:
        """Exposes the :param f: attribute set at initialization."""
        return self._preprocess_fn

    @property
    def dispatch_fn(self) -> Callable[[NodeDType], BatchDType]:
        """Exposes the :param f: attribute set at initialization."""
        return self._dispatch_fn
