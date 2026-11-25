"""Serializable group encoders built on :class:`modelbinder.registry.Component`.

Each encoder is a frozen model of plain scalars -- no callables -- so it pickles trivially, compares by
value, AND exports a portable ``{type, version, **fields}`` spec via ``.to_spec()`` (round-tripped with
``GroupEncoderConfig.from_spec``). ``build`` returns a transformer exposing ``transform`` (and, for the
functional encoders, ``inverse_transform``).

``GroupEncoderConfig`` is the family base; each config subclasses it, is registered with
``@component(type_id, builds=X)`` and named ``XConfig`` after the transformer ``X`` it builds. Add a new encoder the same way.

The categorical encoders (:class:`LabelEncoderConfig`, :class:`OneHotEncoderConfig`) take their vocabulary
explicitly, so a config maps the same way whatever data it is built on: one cell per group in the loaders,
or new data after a reload. Unknown categories always **raise**.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Annotated

import numpy as np
from pydantic import AfterValidator, Field
from modelbinder.registry import Component, component
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, OneHotEncoder

from sckitflow._types import TargetCovariatesEncoderCls

__all__ = [
    "GroupEncoderConfig",
    "GroupEncoderContext",
    "LabelEncoderConfig",
    "OneHotEncoderConfig",
    "IdentityTransformer",
    "IdentityTransformerConfig",
    "Log1pTransformer",
    "Log1pTransformerConfig",
    "AffineTransformer",
    "AffineTransformerConfig",
]


def _unique(vocabulary: tuple[Category, ...]) -> tuple[Category, ...]:
    if len(set(vocabulary)) != len(vocabulary):
        duplicates = sorted(str(v) for v, n in Counter(vocabulary).items() if n > 1)
        raise ValueError(f"The vocabulary lists {duplicates} more than once.")
    return vocabulary


#: A category as it appears in ``.obs``: a string, or an integer or boolean id.
Category = str | int | bool
#: A non-empty category list, each category once.
Vocabulary = Annotated[tuple[Category, ...], Field(min_length=1), AfterValidator(_unique)]


@dataclass(frozen=True)
class GroupEncoderContext:
    """Fit-time inputs for a group encoder -- deliberately NOT part of the serialized config."""

    data: np.ndarray


class GroupEncoderConfig(Component):
    """Family base for serializable group encoders. ``build(context)`` returns the fitted transformer."""

    def build(self, context: GroupEncoderContext) -> TargetCovariatesEncoderCls:
        raise NotImplementedError


@component("group_encoder.label", builds=LabelEncoder)
class LabelEncoderConfig(GroupEncoderConfig):
    """Integer-codes a categorical column.

    :param classes: The vocabulary. Each value's code is its position here, so ``("b", "a")`` codes ``b`` as 0.
        ``transform`` raises on any value not listed.
    :type classes: class: `tuple[str, ...]`
    """

    classes: Vocabulary

    def build(self, context: GroupEncoderContext) -> LabelEncoder:
        encoder = LabelEncoder()
        # Set rather than fit: fitting sorts. An object array is mapped by lookup, so the given order holds.
        encoder.classes_ = np.asarray(self.classes, dtype=object)
        return encoder


@component("group_encoder.one_hot", builds=OneHotEncoder)
class OneHotEncoderConfig(GroupEncoderConfig):
    """One-hot encodes a categorical column.

    :param categories: The vocabulary, in column order. Fit raises if the data holds a value outside it, and
        ``transform`` raises on unknown categories.
    :type categories: class: `tuple[str, ...]`
    """

    categories: Vocabulary

    def build(self, context: GroupEncoderContext) -> OneHotEncoder:
        return OneHotEncoder(categories=[list(self.categories)], handle_unknown="error").fit(
            np.asarray(context.data).reshape(-1, 1)
        )


class IdentityTransformer(FunctionTransformer):
    """Passes values through unchanged."""

    def __init__(self) -> None:
        super().__init__(check_inverse=False)


class Log1pTransformer(FunctionTransformer):
    """``log1p``, inverted by ``expm1``."""

    def __init__(self) -> None:
        super().__init__(func=np.log1p, inverse_func=np.expm1, check_inverse=False)


class AffineTransformer(FunctionTransformer):
    """``x * scale + shift``, inverted by ``(x - shift) / scale``."""

    def __init__(self, scale: float = 1.0, shift: float = 0.0) -> None:
        self.scale, self.shift = scale, shift
        super().__init__(func=self._forward, inverse_func=self._inverse, check_inverse=False)

    def _forward(self, x: np.ndarray) -> np.ndarray:
        return x * self.scale + self.shift

    def _inverse(self, x: np.ndarray) -> np.ndarray:
        return (x - self.shift) / self.scale


@component("group_encoder.identity", builds=IdentityTransformer)
class IdentityTransformerConfig(GroupEncoderConfig):
    """Passes the column through unchanged."""

    def build(self, context: GroupEncoderContext) -> IdentityTransformer:
        return IdentityTransformer().fit(context.data)


@component("group_encoder.log1p", builds=Log1pTransformer)
class Log1pTransformerConfig(GroupEncoderConfig):
    """Applies ``log1p`` (inverse ``expm1``) to a continuous column."""

    def build(self, context: GroupEncoderContext) -> Log1pTransformer:
        return Log1pTransformer().fit(context.data)


@component("group_encoder.affine", builds=AffineTransformer)
class AffineTransformerConfig(GroupEncoderConfig):
    """Scales and shifts a continuous column (``x * scale + shift``).

    :param scale: Multiplicative factor. Defaults to ``1.0``.
    :type scale: class: `float`

    :param shift: Additive offset. Defaults to ``0.0``.
    :type shift: class: `float`
    """

    scale: float = 1.0
    shift: float = 0.0

    def build(self, context: GroupEncoderContext) -> AffineTransformer:
        return AffineTransformer(scale=self.scale, shift=self.shift).fit(context.data)
