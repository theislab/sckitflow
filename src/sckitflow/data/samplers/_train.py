from __future__ import annotations
import abc

from typing import TYPE_CHECKING

from sckitflow._constants import DEFAULT_BATCH_SIZE, DEFAULT_N_GROUPS, MAX_ITER_STEPS
from sckitflow.data._mixins import MappedTree
from sckitflow.data.samplers._base import FSampler, MSampler, BaseSampler

if TYPE_CHECKING:
    from sckitflow.core._types import StepData

__all__ = ["TrainSampler", "FTrainSampler", "MTrainSampler"]


class TrainSampler(BaseSampler):
    """Abstract class for train samplers.
    
    Subclasses need to override the :method sample: method.
    """

    def __init__(
        self,
        tree: MappedTree,
        *args,
        batch_size: int = DEFAULT_BATCH_SIZE,
        n_nodes: int = DEFAULT_N_GROUPS,
        replace_samples: bool = False,
        replace_nodes: bool = False,
        use_nodes_weights: bool = True,
        max_iter_steps: int = MAX_ITER_STEPS,
        **kwargs,
    ) -> None:
        """Initializes the training sampler.

        :param tree: Tree storing the split and matched subpopulations.
        :type tree: class: `MappedTree`

        :param batch_size: The number of observations to sample for each node in a batch.
            Defaults to :constant sckitflow._constants.DEFAULT_BATCH_SIZE:.
        :type batch_size: class: `int`

        :param n_nodes: The number of nodes to sample for each batch.
            Defaults to :constant sckitflow._constants.DEFAULT_N_GROUPS:.
        :type n_nodes: class: `int`

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
            tree,
            *args,
            replace_samples=replace_samples,
            replace_nodes=replace_nodes,
            use_nodes_weights=use_nodes_weights,
            **kwargs,  # forwards e.g. `dispatch_fn` down the MRO to FSampler
        )
        self._batch_size = batch_size
        self._n_nodes = n_nodes
        self._max_iter_steps = max_iter_steps

        self._current_iter_step = 0

    @abc.abstractmethod
    def sample(self) -> tuple[StepData]:
        """Samples a batch of data from the tree.
        
        Must be overridden by derived classes.
        """

    def __iter__(self):
        return self

    def __next__(self):
        if self._current_iter_step < self._max_iter_steps:
            obj = self.sample()
            self._current_iter_step += 1
            return obj
        else:
            raise StopIteration

    @property
    def batch_size(self) -> int:
        """Exposes the :param batch_size: attribute set at initialization."""
        return self._batch_size

    @property
    def n_nodes(self) -> int:
        """Exposes the :param n_nodes: attribute set at initialization."""
        return self._n_nodes

class FTrainSampler(TrainSampler, FSampler):
    """Concrete train sampler using an input callable to process the batch."""
    def sample(self) -> tuple[StepData]:
        """Samples a batch of data from the tree."""
        return self._sample(self._n_nodes, self._batch_size)

class MTrainSampler(TrainSampler, MSampler):
    """Concrete train sampler using an input callable to process the batch."""
    def sample(self) -> tuple[StepData]:
        """Samples a batch of data from the tree."""
        return self._sample(self._batch_size)
