from collections.abc import Iterator
from itertools import chain, islice

from sckitflow._constants import DEFAULT_BATCH_SIZE, DEFAULT_N_GROUPS
from sckitflow.data._abc import DataT, DataTreeT, MatchedDistributionsT
from sckitflow.data.samplers._base import FSampler, Sampler

__all__ = ["TrainSampler", "FTrainSampler"]


class TrainSampler(Sampler[MatchedDistributionsT, DataT]):
    """Abstract class for train samplers."""

    def __init__(
        self,
        tree: DataTreeT,
        *args,
        batch_size: int = DEFAULT_BATCH_SIZE,
        n_nodes: int = DEFAULT_N_GROUPS,
        replace_samples: bool = False,
        replace_nodes: bool = False,
        use_nodes_weights: bool = True,
        max_iter_steps: int | None = None,
        **kwargs,
    ) -> None:
        """Initializes the training sampler.

        :param tree: Tree storing the split and matched subpopulations.
        :type tree: class: `DataTreeT`

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

        :param max_iter_steps: Upper bound on the number of nodes yielded by
            :meth:`__iter__`. `None` (the default) leaves the stream unbounded, which is
            what training expects: the number of steps is set on the trainer
            (``max_steps``) rather than on the sampler. Set it only when iterating the
            sampler standalone.
        :type max_iter_steps: class: `int | None`
        """
        super().__init__(
            tree,
            *args,
            replace_samples=replace_samples,
            replace_nodes=replace_nodes,
            use_nodes_weights=use_nodes_weights,
        )
        self._batch_size = batch_size
        self._n_nodes = n_nodes
        self._max_iter_steps = max_iter_steps

    def sample(self) -> tuple[DataT]:
        """Samples a round of :attr: `n_nodes` nodes, each holding `batch_size` observations."""
        return self._sample(self._n_nodes, self._batch_size)

    def __iter__(self) -> Iterator[DataT]:
        """Yields one node at a time, drawing a fresh round once the current one is spent.

        A node is the unit of training: one node in, one optimizer step out. Nodes are
        still drawn a round of :attr: `n_nodes` at a time so that
        :attr: `replace_nodes` keeps meaning "distinct nodes within a round".

        The stream is re-iterable -- each call builds a fresh generator -- and unbounded
        unless :attr: `max_iter_steps` says otherwise.
        """
        # `iter(callable, sentinel)` keeps calling `sample` (which never returns the
        # sentinel), so rounds are drawn lazily, one only once the previous is spent.
        nodes = chain.from_iterable(iter(self.sample, None))
        if self._max_iter_steps is None:
            yield from nodes
        else:
            yield from islice(nodes, self._max_iter_steps)

    @property
    def batch_size(self) -> int:
        """Exposes the :param batch_size: attribute set at initialization."""
        return self._batch_size

    @property
    def n_nodes(self) -> int:
        """Exposes the :param n_nodes: attribute set at initialization."""
        return self._n_nodes

    @property
    def max_iter_steps(self) -> int | None:
        """Exposes the :param max_iter_steps: attribute set at initialization."""
        return self._max_iter_steps


class FTrainSampler(TrainSampler, FSampler):
    """Concrete train sampler using an input callable to process the batch."""
