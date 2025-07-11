from collections.abc import Sequence

import torch

__all__ = [
    "TorchPRNGKey",
]


class TorchPRNGKey:
    """Object for PRNG keys."""

    def __init__(
        self,
        seed: int | None = None,
    ) -> None:
        """Initializes the key with a given seed.

        :param seed: The seed used to initialize the PRNGKey.
        :type seed: class: `int`
        """
        self._seed = seed

    def split(
        self,
        num: int,
    ) -> Sequence[torch.Generator]:
        """"""  # noqa

        raise NotImplementedError

    def get_generator(
        self,
    ) -> torch.Generator:
        """Returns the generator.

        For the moment, it only returns the global default generator of pytorch.
        """
        return torch.random.default_generator
