from collections.abc import Iterable, Iterator

from sc_flow._constants import DEFAULT_MAX_N_OBS, DEFAULT_N_GROUPS
from sc_flow.data._abc import DataT, DataTreeT, MatchedDistributionsT
from sc_flow.data.samplers._base import FSampler, Sampler

__all__ = ["ValidationSampler", "FValidationSampler"]


class ValidationSampler(Sampler, Iterable):
    """Abstract class for validation samplers."""

    def __init__(
        self,
        tree: DataTreeT,
        *args,
        max_n_obs: int = DEFAULT_MAX_N_OBS,
        n_nodes: int = DEFAULT_N_GROUPS,
        replace_samples: bool = False,
        replace_nodes: bool = False,
        use_nodes_weights: bool = True,
        **kwargs,
    ) -> None:
        """Initializes the training sampler.

        :param tree: Tree storing the split and matched subpopulations.
        :type tree: class: `DataTreeT`

        :param max_n_obs: The maximum number of observations to sample for each node in a batch.
            Defaults to :constant sc_flow._constants.DEFAULT_BATCH_SIZE:.
        :type max_n_obs: class: `int`

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
        self._max_n_obs = max_n_obs
        self._n_nodes = n_nodes
        self._data = self._register_data()

    def _register_data(self) -> tuple[DataT]:
        """Pre-registres the samples for validation."""
        return self._sample(self._n_nodes, self._max_n_obs)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: slice) -> tuple[DataT]:
        return self._data[idx]

    def __iter__(self) -> Iterator[DataT]:
        yield from self._data

    @property
    def max_n_obs(self) -> int:
        """Exposes the :param max_n_obs: attribute set at initialization."""
        return self._max_n_obs

    @property
    def n_nodes(self) -> int:
        """Exposes the :param n_nodes: attribute set at initialization."""
        return self._n_nodes

    @property
    def data(self) -> tuple[MatchedDistributionsT]:
        """Returns the sequence of pre-registered samples."""
        return self._data


class FValidationSampler(ValidationSampler, FSampler):
    """Concrete validation sampler using an input callable to process the batch."""
