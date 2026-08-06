from __future__ import annotations

import abc
from collections.abc import Callable
from functools import cached_property, partial
from typing import TYPE_CHECKING

import numpy as np

from sckitflow._constants import DEFAULT_BATCH_SIZE, DEFAULT_N_GROUPS
from sckitflow.data._composite import MatchedData
from sckitflow.data._mixins import MappedTree
from sckitflow.data.containers._distribution import DistributionData

if TYPE_CHECKING:
    # Imported for typing only; importing at runtime would create a cycle
    # (``core`` depends on ``data``). Samplers never build ``StepData`` themselves --
    # the conversion is injected as ``dispatch_fn`` by the caller (e.g. ``Model``).
    from sckitflow.core._types import StepData

__all__ = ["BaseSampler", "SamplerStochastic", "SamplerSequential", "FSampler", "MSampler"]

class BaseSampler(abc.ABC):
    """Abstract base class for sampler objects.

    Subclasses need to override the :method _dispatch_node: and :method _sample method.
    """

    def __init__(
        self,
        tree: MappedTree,
        *args,
        replace_samples: bool = False,
        **kwargs,
    ) -> None:
        """Initializes the sampler.

        :param tree: Tree storing the split and matched subpopulations.
        :type tree: class: `MappedTree`

        :param replace_samples: Whether to sample observations with replacement
            from each node. Defaults to `False`.
        :type replace_samples: class: `bool`
        """
        self._tree = tree
        self._replace_samples = replace_samples

    @abc.abstractmethod
    def _dispatch_node(self, node: MatchedData) -> StepData:
        """Method used to dispatch a batch of samples from a node.

        Must be overridden by derived classes.

        :param node: The input node to dispatch.
        :type node: class: `MatchedData`
        """

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

    def _sample_from_distr(self, distr: DistributionData, batch_size: int = DEFAULT_BATCH_SIZE) -> DistributionData:
        """Samples a batch of observations from a distribution object.

        :param distr: The distribution object to sample from.
        :type distr: class: `DistributionDType`

        :param batch_size: The number of observations to be sampled in the batch.
            Defaults to :constant: `sckitflow.constants.DEFAULT_BATCH_SIZE`.
        :type batch_size: class: `int`
        """
        mask = self._sample_indices(len(distr), batch_size)
        return distr[mask]

    def _sample_observations(
        self,
        node: MatchedData,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> StepData:
        """Samples a batch of observations from a node of matched distributions.

        :param node: The node object to sample from.
        :type node: class: `MatchedData`

        :param batch_size: The number of observations to be sampled in the batch.
            It will be the same for both source and target distributions.
            Defaults to :constant: `sckitflow.constants.DEFAULT_BATCH_SIZE`.
        :type batch_size: class: `int`
        """
        target_distr = self._sample_from_distr(node["target"], batch_size)
        source = node.get("source")
        if source is not None:
            source_distr = self._sample_from_distr(source, batch_size)
        else:
            source_distr = None
        batch_data = MatchedData(target=target_distr, source=source_distr)
        return self._dispatch_node(batch_data)

    @abc.abstractmethod
    def _sample(
        self,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> tuple[StepData] | tuple[list[StepData]]:
        """Method used to sample a batch of data from the tree.

        Must be overridden by derived classes.

        :param batch_size: The number of observations to load in the batch.
        :type batch_size: class: `int`
        """

    @property
    def tree(self) -> MappedTree:
        """Returns the underlying data tree."""
        return self._tree

    @property
    def replace_samples(self) -> bool:
        """Exposes the :param replace_samples: attribute set at initialization."""
        return self._replace_samples

    @cached_property
    def flattened_data(self) -> tuple[MatchedData]:
        """Caches the flattened array of leaf nodes of the data tree.

        The transformation defined in :method: `self._preprocess_node` is applied upon caching.
        """
        return self._tree.flatten()

class SamplerStochastic(BaseSampler[MatchedData, StepData]):
    """Abstract base class for sampler objects on trees.

    Subclasses need to override the :method _dispatch_node: method.
    """

    def __init__(
        self,
        tree: MappedTree,
        *args,
        replace_samples: bool = False,
        replace_nodes: bool = False,
        use_nodes_weights: bool = True,
        inverse_frequency_weights: bool = True,
        **kwargs,
    ) -> None:
        """Initializes the sampler.

        :param tree: Tree storing the split and matched subpopulations.
        :type tree: class: `MappedTree`

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

        :param inverse_frequency_weights: Whether to sample nodes according to their inverse
            frequency. Only used when :param: `use_node_weights` is `True`. Defaults to `True`.
        :type inverse_frequency_weights: class: `bool`
        """
        super().__init__(tree, *args, replace_samples=replace_samples, **kwargs)
        self._replace_nodes = replace_nodes
        self._use_nodes_weights = use_nodes_weights
        self._inverse_frequency_weights = inverse_frequency_weights

    def _compute_nodes_freq(self) -> np.ndarray:
        """Returns the array of relative frequencies of each node.

        The relative frequency is computed by taking into account the number of observations
        in the target distribution.
        """
        counts = np.array([len(e.target) for e in self.flattened_data])
        return counts / counts.sum()

    def _sample_nodes(
        self,
        n_nodes: int,
    ) -> np.ndarray[MatchedData]:
        """Samples an array of leaf nodes from the tree.

        :param n_nodes: The number of nodes to sample.
        :type n_nodes: class: `int`
        """
        return np.random.choice(
            self.flattened_data,
            n_nodes,
            p=self.nodes_p if self._use_nodes_weights else None,
            replace=self._replace_nodes,
        )

    def _sample(
        self,
        n_nodes: int = DEFAULT_N_GROUPS,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> tuple[StepData]:
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
    def replace_nodes(self) -> bool:
        """Exposes the :param replace_nodes: attribute set at initialization."""
        return self._replace_nodes

    @property
    def use_nodes_weights(self) -> bool:
        """Exposes the :param use_nodes_weights: attribute set at initialization."""
        return self._use_nodes_weights

    @cached_property
    def nodes_p(self) -> np.ndarray:
        """Computes the nodes probabilities for sampling.

        When :attr: `inverse_frequence_weights` is `False`, the probabilities will be simply given by the
        frequency. Otherwise, normalized inverse frequency is used.
        """
        nodes_freq = self._compute_nodes_freq()
        if self._inverse_frequency_weights:
            nodes_inv_freq = 1 / nodes_freq
            return nodes_inv_freq / nodes_inv_freq.sum()
        return nodes_freq

    @property
    def inverse_frequency_weights(self) -> bool:
        """Exposes to :param: `inverse_frequency_weights` set at initialization."""
        return self._inverse_frequency_weights

class FSampler(SamplerStochastic[MatchedData, StepData]):
    """Concrete class using an input callable to process the batch."""

    def __init__(
        self,
        data: MappedTree,
        dispatch_fn: Callable[[MatchedData], StepData] | None = None,
        replace_samples: bool = False,
        replace_nodes: bool = False,
        use_nodes_weights: bool = True,
        **kwargs,
    ) -> None:
        """Initializes the sampler.

        :param tree: Tree storing the split and matched subpopulations.
        :type tree: class: `MappedTree`

        :param dispatch_fn: The function used to post-process the batch of matched data
            before being dispatched. It will be used to override the corresponding
            method of the abstract class by wrapping it around this callable.
        :type dispatch_fn: class: `Callable[[MatchedData], StepData]`

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
        self._dispatch_fn = (lambda x: x) if dispatch_fn is None else dispatch_fn

        super().__init__(
            data, replace_samples=replace_samples, replace_nodes=replace_nodes, use_nodes_weights=use_nodes_weights
        )

    def _dispatch_node(self, node: MatchedData) -> StepData:
        """Method used to dispatch a batch of samples from a node.

        The dispatch function set at initialization with :param: `dispatch_fn` will be used.

        :param node: The input node to dispatch.
        :type node: class: `MatchedData`
        """
        return self._dispatch_fn(node)

    @property
    def dispatch_fn(self) -> Callable[[MatchedData], StepData]:
        """Exposes the :param f: attribute set at initialization."""
        return self._dispatch_fn

class SamplerSequential(BaseSampler[MatchedData, StepData]):
    """Abstract base class for sampler objects on trees
    which uses all nodes.

    Subclasses need to override the :method _dispatch_node: method.
    """

    def _sample(
        self,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> tuple[list[StepData]]:
        """Samples a batch of data from the tree.

        Sampling is done by taking all nodes in order.

        :param batch_size: The number of observations to load in the batch.
        :type batch_size: class: `int`
        """
        nodes = self.flattened_data
        sample_fn = partial(self._sample_observations, batch_size=batch_size)
        return (list(map(sample_fn, nodes)), )

class MSampler(SamplerSequential[MatchedData, StepData]):
    """Concrete class using an input callable to process the batch."""

    def __init__(
        self,
        data: MappedTree,
        dispatch_fn: Callable[[MatchedData], StepData] | None = None,
        replace_samples: bool = False,
        **kwargs,
    ) -> None:
        """Initializes the sampler.

        :param tree: Tree storing the split and matched subpopulations.
        :type tree: class: `MappedTree`

        :param dispatch_fn: The function used to post-process the batch of matched data
            before being dispatched. It will be used to override the corresponding
            method of the abstract class by wrapping it around this callable.
        :type dispatch_fn: class: `Callable[[MatchedData], StepData]`

        :param replace_samples: Whether to sample observations with replacement
            from each node. Defaults to `False`.
        :type replace_samples: class: `bool`
        """
        self._dispatch_fn = (lambda x: x) if dispatch_fn is None else dispatch_fn

        super().__init__(data, replace_samples=replace_samples)

    def _dispatch_node(self, node: MatchedData) -> StepData:
        """Method used to dispatch a batch of samples from a node.

        The dispatch function set at initialization with :param: `dispatch_fn` will be used.

        :param node: The input node to dispatch.
        :type node: class: `MatchedData`
        """
        return self._dispatch_fn(node)

    @property
    def dispatch_fn(self) -> Callable[[MatchedData], StepData]:
        """Exposes the :param f: attribute set at initialization."""
        return self._dispatch_fn