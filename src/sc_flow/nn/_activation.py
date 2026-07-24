from __future__ import annotations

from typing import Literal

import torch

__all__ = ["ActivationId", "resolve_activation"]

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


def resolve_activation(
    activation: ActivationId | type[torch.nn.Module] | None,
    default: ActivationId,
) -> type[torch.nn.Module]:
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
