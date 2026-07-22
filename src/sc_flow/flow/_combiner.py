import abc
from typing import Any

import torch

from sc_flow._constants import DEFAULT_COMBINER
from sc_flow._types import CombinerId
from sc_flow._utils import verify_fn_kwargs_dictionary
from sc_flow.core._torch_utils import make_concatenation_possible
from sc_flow.core.nn._modules import BaseModule, Resnet1d

__all__ = [
    "COMBINER_REGISTRY",
    "BaseCombiner",
    "ConcatCombiner",
    "Resnet1dCombiner",
    "get_combiner",
    "register_combiner",
]

#: Built-in + extension-registered combiners, keyed by string id (the on-disk name saved to config.json).
COMBINER_REGISTRY: dict[str, type["BaseCombiner"]] = {}


def register_combiner(combiner_id: str):
    """Register a :class:`BaseCombiner` subclass under a string ``combiner_id``.

    The id is what a user passes as ``combiner=...`` and what is written to ``config.json`` (so it must be
    stable and globally unique). An extension can register its own combiner from its own package — no edit
    to sc-flow — and refer to it by id; a checkpoint that names an unregistered id fails loudly at load
    (the providing package likely wasn't imported). Registered combiners must accept the standard signature
    ``(latent_state_dim, latent_time_dim, latent_condition_dim=None, **combiner_kwargs)`` — the velocity
    field supplies the dims, ``combiner_kwargs`` (JSON, from config) carries any hyperparameters.
    """

    def _decorator(cls: type["BaseCombiner"]) -> type["BaseCombiner"]:
        existing = COMBINER_REGISTRY.get(combiner_id)
        if existing is not None and existing is not cls:
            msg = f"Combiner id {combiner_id!r} is already registered to {existing.__name__}."
            raise ValueError(msg)
        COMBINER_REGISTRY[combiner_id] = cls
        return cls

    return _decorator


class BaseCombiner(BaseModule):
    """Base class for combiners.

    A combiner fuses the encoded state with the conditioning signals — the encoded time, and the
    condition embedding when present — into the vector fed to the velocity-field decoder.
    """

    def __init__(
        self,
        latent_state_dim: int,
        latent_time_dim: int,
        latent_condition_dim: int | None = None,
    ) -> None:
        """Initializes the combiner.

        :param latent_state_dim: The latent dimensionality of the input states.
        :type latent_state_dim: class: `int`

        :param latent_time_dim: The latent dimensionality of the input time index.
        :type latent_time_dim: class: `int`

        :param latent_condition_dim: (Optional) Dimensionality of the condition embedding fused with the
            encoded state and time (the trailing dim of ``encoded_condition``, i.e. the condition-encoder
            output, plus the source-encoder output when present). ``None`` for an unconditional combiner,
            where only state and time are combined.
        :type latent_condition_dim: class: `int | None`
        """
        super().__init__()
        self._latent_state_dim = latent_state_dim
        self._latent_time_dim = latent_time_dim
        self._latent_condition_dim = latent_condition_dim

    @property
    @abc.abstractmethod
    def output_dim(
        self,
    ) -> int:
        """Returns the dimensionality of the combined output."""

    @abc.abstractmethod
    def forward(
        self,
        encoded_t: torch.Tensor,
        encoded_state: torch.Tensor,
        encoded_condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """The combined representation fed to the velocity-field decoder.

        :param encoded_t: The time index at which the velocity field is computed.
        :type encoded_t: class: `torch.Tensor`

        :param encoded_state: The state at which the velocity field is computed.
        :type encoded_state: class: `torch.Tensor`

        :param encoded_condition: (Optional) The condition embedding to fuse with the encoded state and time.
            Its trailing dimension must match :attr: `self._latent_condition_dim`,
            otherwise a :class: `RuntimeError` is raised.
        :type encoded_condition: class: `torch.Tensor`
        """

    def _verify_inputs_shape(
        self,
        encoded_t: torch.Tensor,
        encoded_state: torch.Tensor,
        encoded_condition: torch.Tensor | None = None,
    ) -> None:
        """Verifies that the inputs match their respective expected shapes."""
        # sanity checks
        if encoded_state.shape[-1] != self._latent_state_dim:
            msg = (
                "The encoded state has the wrong trailing dimension."
                f"Found {encoded_state.shape[-1]}, expected {self._latent_state_dim}"
            )
            raise RuntimeError(msg)
        if encoded_t.shape[-1] != self._latent_time_dim:
            msg = (
                "The encoded time has the wrong trailing dimension."
                f"Found {encoded_t.shape[-1]}, expected {self._latent_time_dim}"
            )
            raise RuntimeError(msg)
        if encoded_condition is not None:
            if encoded_condition.shape[-1] != self._latent_condition_dim:
                msg = (
                    f"Incorrect trailing dimension for additional condition input."
                    f"Found {encoded_condition.shape[-1]}, expected {self._latent_condition_dim}"
                )
                raise RuntimeError(msg)
        else:
            if self._latent_condition_dim is not None:
                msg = (
                    "When the latent condition dimension is specified,"
                    "the corresponding input should be passed, found `None`."
                )
                raise RuntimeError(msg)


@register_combiner("concat")
class ConcatCombiner(BaseCombiner):
    """Class for concatenation based combiner layers."""

    def __init__(
        self,
        latent_state_dim: int,
        latent_time_dim: int,
        latent_condition_dim: int | None = None,
    ) -> None:
        """Initializes the combiner.

        :param latent_state_dim: The latent dimensionality of the input states.
        :type latent_state_dim: class: `int`

        :param latent_time_dim: The latent dimensionality of the input time index.
        :type latent_time_dim: class: `int`

        :param latent_condition_dim: (Optional) Dimensionality of the condition embedding fused with the
            encoded state and time (the trailing dim of ``encoded_condition``, i.e. the condition-encoder
            output, plus the source-encoder output when present). ``None`` for an unconditional combiner,
            where only state and time are combined.
        :type latent_condition_dim: class: `int | None`
        """
        super().__init__(
            latent_state_dim,
            latent_time_dim,
            latent_condition_dim,
        )

        self._identity = self._make_modules()

    def _make_modules(
        self,
    ) -> torch.nn.Module:
        """Initializes the module."""
        return torch.nn.Identity()

    @property
    def output_dim(
        self,
    ) -> int:
        """Return the dimensionality of the combined output."""
        out_dim = self._latent_state_dim + self._latent_time_dim
        if self._latent_condition_dim is not None:
            out_dim = out_dim + self._latent_condition_dim
        return out_dim

    def forward(
        self,
        encoded_t: torch.Tensor,
        encoded_state: torch.Tensor,
        encoded_condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass: combine the encoded state, time, and optional condition into the decoder input.

        :param encoded_t: The encoded representation for the current time index.
            Its trailing dimension should match what specified in the :attr: `self._latent_time_dim`
            attribute, otherwise a :class: `RuntimeError` is raised.
        :type encoded_t: class: `torch.Tensor`

        :param encoded_state: The encoded representation for the current states.
            Its trailing dimension should match what specified in the :attr: `self._latent_state_dim`
            attribute, otherwise a :class: `RuntimeError` is raised.
        :type encoded_state: class: `torch.Tensor`

        :param encoded_condition: (Optional) The condition embedding to fuse with the encoded state and time.
            Its trailing dimension must match :attr: `self._latent_condition_dim`,
            otherwise a :class: `RuntimeError` is raised.
        :type encoded_condition: class: `torch.Tensor`
        """
        # sanity checks
        self._verify_inputs_shape(
            encoded_t,
            encoded_state,
            encoded_condition,
        )

        # concatenating input
        to_concat = (encoded_state, make_concatenation_possible(encoded_t, encoded_state, -1))
        if encoded_condition is not None:
            to_concat = to_concat + (make_concatenation_possible(encoded_condition, encoded_state, -1),)
        concat_input = torch.concatenate(to_concat, dim=-1)
        return self._identity(concat_input)


@register_combiner("resnet1d")
class Resnet1dCombiner(BaseCombiner):
    """Class for residual network based combiner layers.

    Its per-layer hyperparameters go in the ``resnet_kwargs`` dict; via the registry they are passed as
    ``combiner="resnet1d", combiner_kwargs={"resnet_kwargs": {...}}``.
    """

    def __init__(
        self,
        latent_state_dim: int,
        latent_time_dim: int,
        latent_condition_dim: int | None = None,
        resnet_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initializes the resnet based combiner.

        :param latent_state_dim: The latent dimensionality of the input states.
        :type latent_state_dim: class: `int`

        :param latent_time_dim: The latent dimensionality of the input time index.
        :type latent_time_dim: class: `int`

        :param latent_condition_dim: (Optional) Dimensionality of the condition embedding fused with the
            encoded state and time (the trailing dim of ``encoded_condition``, i.e. the condition-encoder
            output, plus the source-encoder output when present). ``None`` for an unconditional combiner,
            where only state and time are combined.
        :type latent_condition_dim: class: `int | None`

        :param resnet_kwargs: Additional key-word arguments used to initialize the :class: `Resnet1d` object.
        :type resnet_kwargs: class: `dict[str, Any]`
        """
        super().__init__(
            latent_state_dim,
            latent_time_dim,
            latent_condition_dim,
        )
        self._resnet_kwargs = {} if resnet_kwargs is None else resnet_kwargs

        self._resnet = self._make_modules()

    def _make_modules(
        self,
    ) -> torch.nn.Module:
        """Initializes the module."""
        verify_fn_kwargs_dictionary(Resnet1d.__init__, self._resnet_kwargs)
        return Resnet1d(
            self._latent_state_dim,
            self.embedding_dim,
            **self._resnet_kwargs,
        )

    def forward(
        self,
        encoded_t: torch.Tensor,
        encoded_state: torch.Tensor,
        encoded_condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass: combine the encoded state, time, and optional condition into the decoder input.

        :param encoded_t: The encoded representation for the current time index.
            Its trailing dimension should match what specified in the :attr: `self._latent_time_dim`
            attribute, otherwise a :class: `RuntimeError` is raised.
        :type encoded_t: class: `torch.Tensor`

        :param encoded_state: The encoded representation for the current states.
            Its trailing dimension should match what specified in the :attr: `self._latent_state_dim`
            attribute, otherwise a :class: `RuntimeError` is raised.
        :type encoded_state: class: `torch.Tensor`

        :param encoded_condition: (Optional) The condition embedding to fuse with the encoded state and time.
            Its trailing dimension must match :attr: `self._latent_condition_dim`,
            otherwise a :class: `RuntimeError` is raised.
        :type encoded_condition: class: `torch.Tensor`
        """
        # sanity checks
        self._verify_inputs_shape(
            encoded_t,
            encoded_state,
            encoded_condition,
        )

        # build the [time (+ condition)] conditioning vector the resnet is conditioned on
        to_concat = (make_concatenation_possible(encoded_t, encoded_state, -1),)
        if encoded_condition is not None:
            to_concat = to_concat + (make_concatenation_possible(encoded_condition, encoded_state, -1),)

        conditions = torch.concatenate(to_concat, dim=-1)
        return self._resnet(encoded_state, conditions)

    @property
    def output_dim(
        self,
    ) -> int:
        """Return the dimensionality of the combined output."""
        return self._resnet.output_dim

    @property
    def embedding_dim(
        self,
    ) -> int:
        """Dimensionality of the conditioning embedding (encoded time, plus condition when present) fed to the resnet."""
        embedding_dim = self._latent_time_dim
        if self._latent_condition_dim is not None:
            embedding_dim = embedding_dim + self._latent_condition_dim
        return embedding_dim


def get_combiner(
    latent_state_dim: int,
    latent_time_dim: int,
    latent_condition_dim: int | None = None,
    combiner_id: CombinerId | str | None = None,
    combiner_kwargs: dict[str, Any] | None = None,
) -> BaseCombiner:
    """Build a combiner by its registered ``combiner_id``.

    The velocity field supplies the latent dims; ``combiner_kwargs`` (JSON-serializable, restored from
    ``config.json``) carries any hyperparameters. Raises if ``combiner_id`` is not registered — e.g. an
    extension that provides it was not imported.
    """
    combiner_id = DEFAULT_COMBINER if combiner_id is None else combiner_id
    cls = COMBINER_REGISTRY.get(combiner_id)
    if cls is None:
        msg = (
            f"Combiner {combiner_id!r} is not registered. Available: {sorted(COMBINER_REGISTRY)}. "
            f"(If it is provided by an extension, import that package before building/loading the model.)"
        )
        raise ValueError(msg)
    combiner_kwargs = {} if combiner_kwargs is None else combiner_kwargs
    return cls(latent_state_dim, latent_time_dim, latent_condition_dim=latent_condition_dim, **combiner_kwargs)
