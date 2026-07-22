"""Serializable activation choice.

An activation is a *parameter-free* architecture choice: it carries no weights and comes from a small,
settled set. Per ``docs/plans/state.md`` §10 such choices are serialized but stay a **validated enum**, not
a discriminated component spec — there is nothing to register and no third-party extension surface worth
opening. This module is the enum + its resolver.

Storing the id (a string) rather than the ``torch.nn.Module`` *class* is what closes the silent-default
footgun (§14): a class cannot live in a JSON config, so auto-capture would drop it and rebuild the default,
turning a saved ``Tanh`` into a ``ReLU`` that still matches every ``state_dict`` key. A string id round-trips
exactly.
"""

from __future__ import annotations

from typing import Literal

import torch

__all__ = ["ActivationId", "resolve_activation", "activation_id"]

#: The closed set of built-in activations, by stable string id.
ActivationId = Literal[
    "relu",
    "leaky_relu",
    "elu",
    "gelu",
    "silu",
    "tanh",
    "sigmoid",
    "softplus",
    "identity",
]

_ACTIVATIONS: dict[str, type[torch.nn.Module]] = {
    "relu": torch.nn.ReLU,
    "leaky_relu": torch.nn.LeakyReLU,
    "elu": torch.nn.ELU,
    "gelu": torch.nn.GELU,
    "silu": torch.nn.SiLU,
    "tanh": torch.nn.Tanh,
    "sigmoid": torch.nn.Sigmoid,
    "softplus": torch.nn.Softplus,
    "identity": torch.nn.Identity,
}
_CLASS_TO_ID: dict[type[torch.nn.Module], str] = {cls: name for name, cls in _ACTIVATIONS.items()}


def resolve_activation(
    activation: ActivationId | type[torch.nn.Module] | None,
    default: ActivationId,
) -> type[torch.nn.Module]:
    """Resolve an activation to a ``torch.nn.Module`` class.

    Accepts a built-in string id, a ``torch.nn.Module`` subclass (the runtime-only research path — not
    portable), or ``None`` (use ``default``). A raw class is passed through unchanged so experimentation is
    unhindered; only the *config* layer insists on a string id.
    """
    if activation is None:
        activation = default
    if isinstance(activation, str):
        try:
            return _ACTIVATIONS[activation]
        except KeyError:
            raise ValueError(
                f"Unknown activation id {activation!r}; built-in ids are {sorted(_ACTIVATIONS)}."
            ) from None
    if isinstance(activation, type) and issubclass(activation, torch.nn.Module):
        return activation
    raise TypeError(f"activation must be an id string, a torch.nn.Module subclass, or None; found {activation!r}.")


def activation_id(activation: ActivationId | type[torch.nn.Module] | None, default: ActivationId) -> ActivationId:
    """The stable string id for an activation, for serialization.

    ``None`` maps to ``default``; a string id is validated; a built-in class maps back to its id. A custom
    (unregistered) class has no portable id and raises — the caller must reject it before export.
    """
    if activation is None:
        return default
    if isinstance(activation, str):
        if activation not in _ACTIVATIONS:
            raise ValueError(f"Unknown activation id {activation!r}; built-in ids are {sorted(_ACTIVATIONS)}.")
        return activation  # type: ignore[return-value]
    if isinstance(activation, type) and activation in _CLASS_TO_ID:
        return _CLASS_TO_ID[activation]  # type: ignore[return-value]
    raise ValueError(
        f"Activation {activation!r} has no portable id (only built-in activations are serializable): "
        f"built-in ids are {sorted(_ACTIVATIONS)}."
    )
