from sc_flow._constants import DEFAULT_BATCH_SIZE, DEFAULT_N_GROUPS
from sc_flow.data._abc import DataT, DataTreeT
from sc_flow.data.samplers._base import FSampler, Sampler

__all__ = ["TrainSampler", "FTrainSampler"]


class TrainSampler(Sampler):
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
        **kwargs,
    ) -> None:
        """Initializes the training sampler.

        :param tree: Tree storing the split and matched subpopulations.
        :type tree: class: `DataTreeT`

        :param batch_size: The number of observations to sample for each node in a batch.
            Defaults to :constant sc_flow._constants.DEFAULT_BATCH_SIZE:.
        :type batch_size: class: `int`

        :param n_nodes: The number of nodes to sample for each batch.
            Defaults to :constant sc_flow._constants.DEFAULT_N_GROUPS:.
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
        )
        self._batch_size = batch_size
        self._n_nodes = n_nodes

    def sample(self) -> tuple[DataT]:
        """Samples a batch of data from the tree."""
        return self._sample(self._n_nodes, self._batch_size)

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
