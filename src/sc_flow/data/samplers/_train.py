import numpy as np

from sc_flow._constants import DEFAULT_BATCH_SIZE, DEFAULT_N_GROUPS
from sc_flow.data.samplers._base import BatchDType, FSampler, Sampler, TreeDType

__all__ = ["TrainSampler", "FTrainSampler"]


class TrainSampler(Sampler):
    """"""  # noqa

    def __init__(
        self,
        tree: TreeDType,
        *args,
        batch_size: int = DEFAULT_BATCH_SIZE,
        n_nodes: int = DEFAULT_N_GROUPS,
        replace_samples: bool = False,
        replace_nodes: bool = False,
        use_groups_weights: bool = True,
    ) -> None:
        """"""  # noqa
        super().__init__(
            tree,
            *args,
            replace_samples=replace_samples,
            replace_nodes=replace_nodes,
            use_groups_weights=use_groups_weights,
        )
        self._batch_size = batch_size
        self._n_nodes = n_nodes

    def sample(self) -> np.ndarray[BatchDType]:
        """"""  # noqa
        sample = self._sample(self._n_nodes, self._batch_size)
        return self._dispatch_sample(sample)

    @property
    def batch_size(self) -> int:
        """"""  # noqa
        return self._batch_size

    @property
    def n_nodes(self) -> int:
        """"""  # noqa
        return self._n_nodes


class FTrainSampler(TrainSampler, FSampler): ...
