from collections.abc import Iterable, Iterator

import numpy as np

from sc_flow._constants import DEFAULT_MAX_N_OBS, DEFAULT_N_GROUPS
from sc_flow.data.samplers._base import BatchDType, FSampler, Sampler, TreeDType

__all__ = ["ValidationSampler", "FValidationSampler"]


class ValidationSampler(Sampler, Iterable):
    """"""  # noqa

    def __init__(
        self,
        tree: TreeDType,
        *args,
        max_n_obs: int = DEFAULT_MAX_N_OBS,
        n_nodes: int = DEFAULT_N_GROUPS,
        replace_samples: bool = False,
        replace_nodes: bool = False,
        use_nodes_weights: bool = True,
    ) -> None:
        """"""  # noqa
        super().__init__(
            tree,
            *args,
            replace_samples=replace_samples,
            replace_nodes=replace_nodes,
            use_nodes_weights=use_nodes_weights,
        )
        self._max_n_obs = max_n_obs
        self._n_nodes = n_nodes
        self._samples = self._register_samples()

    def _register_samples(self) -> np.ndarray[BatchDType]:
        """"""  # noqa
        return self._sample(self._n_nodes, self._max_n_obs)

    def __len__(self) -> int:
        """"""  # noqa
        return len(self._samples)

    def __getitem__(self, idx: slice) -> np.ndarray[BatchDType]:
        """"""  # noqa
        samples = self._samples[idx]
        return self._dispatch_sample(samples)

    def __iter__(self) -> Iterator[BatchDType]:
        for group in self._samples:
            yield self._dispatch_sample(group)

    @property
    def max_n_obs(self) -> int:
        """"""  # noqa
        return self._max_n_obs

    @property
    def n_nodes(self) -> int:
        """"""  # noqa
        return self._n_nodes


class FValidationSampler(ValidationSampler, FSampler): ...
